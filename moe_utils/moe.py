# SPDX-License-Identifier: MIT
"""Standalone MOE dispatch logic for this repository.

Only backend paths implemented by this repository belong here. Runtime MOE_C
operator dispatch is routed through fused_moe_c.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .shuffle import (
    moe_shfl_scale,
    moe_shfl_weight,
    repack_w4a8_weight_contiguous_to_blocked,
    w4a8_moe_layout_shuffle,
    w4a8_moe_repack_shuffle,
    w4a16_marlin_weight_1,
    w4a16_marlin_weight_2,
    w4a16_marlin_scale,
    w8a16_marlin_weight_1,
    w8a16_marlin_weight_2,
    w8a8_moe_layout_shuffle_gemm2,
    w8a8_moe_repack_shuffle,
    wfp4a8_moe_layout_shuffle,
    wfp4a8_moe_repack_shuffle,
)

class MoeSolutionType:
    MOE_C = "moe_c"


class MoeQuantType:
    W16A16 = "w16a16"
    W4A16 = "w4a16"
    WFP4A16 = "fp4_w4a16"
    WFP4A8 = "fp4_w4a8"
    W4A8 = "w4a8"
    W8A8 = "int8_w8a8"
    FP8_W8A8 = "fp8_w8a8"
    INT8_W8A16 = "int8_w8a16"
    FP8_W8A16 = "fp8_w8a16"

@dataclass
class MoeConfig:
    quant_type: Optional[str] = None
    solution_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    need_shuffle: bool = False
    need_shuffle_scale: bool = False


AiterMoeConfig = MoeConfig


def _config_root() -> Path:
    return Path(__file__).resolve().parents[1] / "moe_c_configs"


def _gfx_version() -> str:
    env_arch = os.environ.get("GPU_ARCHS") or os.environ.get("PYTORCH_ROCM_ARCH")
    if env_arch:
        return env_arch.split(";")[0].split(",")[0]
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return "gfx938"


def _load_config_file(path: Path) -> Optional[Dict[int, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return {int(key): value for key, value in json.load(f).items()}


def _moe_c_config_category(dtype: str, block_size: Optional[int]) -> str:
    if dtype == "fp4_w4a8" and block_size:
        return f"fp4_w4a8_groupwise{block_size}"
    return dtype


def _find_moe_c_config(
    E: int,
    N: int,
    K: int,
    dtype: str,
    is_bottom: bool = False,
    block_size: Optional[int] = None,
) -> Optional[Dict[int, Any]]:
    arch = _gfx_version()
    suffix = ",is_bottom=True" if is_bottom else ""
    names = [
        f"E={E},N={N},K={K},dtype={dtype}{suffix}.json",
        f"E={E},N={N},dtype={dtype}{suffix}.json",
    ]
    category = _moe_c_config_category(dtype, block_size)
    roots = [
        _config_root() / arch / category,
        _config_root() / arch / dtype,
        _config_root() / category,
        _config_root() / dtype,
        _config_root(),
    ]
    for root in roots:
        for name in names:
            config = _load_config_file(root / name)
            if config is not None:
                return config
    return None


def _pick_closest_config(configs: Dict[int, Any], m: int) -> Dict[str, Any]:
    key_selected = min(configs.keys(), key=lambda x: abs(x - m))
    config = dict(configs[key_selected])
    config["key_selected"] = key_selected
    return config


def get_moe_config(
    M: int,
    E: int,
    N1: int,
    N2: int,
    K: int,
    top_k: int,
    block_size: Optional[int],
    dtype: torch.dtype,
    quant_type: str,
    **_,
) -> Tuple[bool, MoeConfig]:
    del N2, top_k, dtype

    n = N1 // 2
    dtype_name_by_quant = {
        MoeQuantType.W16A16: "w16a16",
        MoeQuantType.W4A16: "int4_w4a16",
        MoeQuantType.WFP4A16: "fp4_w4a16",
        MoeQuantType.WFP4A8: "fp4_w4a8",
        MoeQuantType.W4A8: "int8_w4a8",
        MoeQuantType.W8A8: "int8_w8a8",
        MoeQuantType.FP8_W8A8: "fp8_w8a8",
        MoeQuantType.INT8_W8A16: "int8_w8a16",
    }
    dtype_name = dtype_name_by_quant.get(quant_type)
    if dtype_name is None:
        raise NotImplementedError(f"Unsupported local MOE quant_type: {quant_type}")

    configs = _find_moe_c_config(
        E=E, N=n, K=K, dtype=dtype_name, is_bottom=False, block_size=block_size
    )
    if configs is None:
        return False, MoeConfig(quant_type=quant_type)

    need_shuffle = quant_type not in (MoeQuantType.W4A16, MoeQuantType.WFP4A16)
    need_shuffle_scale = quant_type in (MoeQuantType.W4A16, MoeQuantType.WFP4A16)

    return True, MoeConfig(
        quant_type=quant_type,
        solution_type=MoeSolutionType.MOE_C,
        config=_pick_closest_config(configs, M),
        need_shuffle=need_shuffle,
        need_shuffle_scale=need_shuffle_scale,
    )


def get_moe_config_bottom(
    M: int,
    E: int,
    N: int,
    K: int,
    quant_type: str,
) -> Dict[str, Any]:
    dtype_name_by_quant = {
        MoeQuantType.W16A16: "w16a16",
        MoeQuantType.W4A16: "int4_w4a16",
        MoeQuantType.WFP4A16: "fp4_w4a16",
        MoeQuantType.WFP4A8: "fp4_w4a8",
        MoeQuantType.W4A8: "int8_w4a8",
        MoeQuantType.W8A8: "int8_w8a8",
        MoeQuantType.FP8_W8A8: "fp8_w8a8",
        MoeQuantType.INT8_W8A16: "int8_w8a16",
    }
    dtype_name = dtype_name_by_quant.get(quant_type)
    if dtype_name is None:
        raise NotImplementedError(f"Unsupported local MOE quant_type: {quant_type}")
    block_size = 32 if quant_type == MoeQuantType.WFP4A8 else None
    configs = _find_moe_c_config(
        E=E, N=N, K=K, dtype=dtype_name, is_bottom=True, block_size=block_size
    )
    if configs is None:
        raise RuntimeError(f"No {dtype_name} bottom config found for E={E}, N={N}, K={K}")
    return _pick_closest_config(configs, M)


def moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_config: MoeConfig,
    inplace: bool = False,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list] = None,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    output_dtype: Optional[torch.dtype] = None,
    **_,
) -> torch.Tensor:
    if moe_config.solution_type == MoeSolutionType.MOE_C:
        from .fused_moe_c import moe_c_fused_experts

        return moe_c_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=inplace,
            activation=activation,
            use_int4_w4a16=moe_config.quant_type == MoeQuantType.W4A16,
            use_fp4_w4a16=moe_config.quant_type == MoeQuantType.WFP4A16,
            use_fp4_w4a8=moe_config.quant_type == MoeQuantType.WFP4A8,
            use_int8_w4a8=moe_config.quant_type == MoeQuantType.W4A8,
            use_int8_w8a8=moe_config.quant_type == MoeQuantType.W8A8,
            use_fp8_w8a8=moe_config.quant_type == MoeQuantType.FP8_W8A8,
            use_int8_w8a16=moe_config.quant_type == MoeQuantType.INT8_W8A16,
            use_w16a16=moe_config.quant_type == MoeQuantType.W16A16,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
            routed_scaling_factor=routed_scaling_factor,
            compute_dtype=output_dtype or hidden_states.dtype,
        )

    raise NotImplementedError(
        f"Local moe currently supports only MOE_C, got solution_type={moe_config.solution_type}"
    )


get_aiter_moe_config = get_moe_config
aiter_moe = moe
aiter_moe_shfl_weight = moe_shfl_weight
aiter_moe_shfl_scale = moe_shfl_scale
