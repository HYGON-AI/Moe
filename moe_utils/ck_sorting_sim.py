from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CkSortingResult:
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    tokens_positions_per_expert: torch.Tensor
    m_indices: torch.Tensor
    token_indices: torch.Tensor
    route_indices: torch.Tensor
    moe_c_sorted_token_ids: torch.Tensor
    expert_offsets: torch.Tensor
    active_expert_mask: torch.Tensor


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _check_topk(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be 2D [M, topk], got {tuple(topk_ids.shape)}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights must have the same shape as topk_ids, "
            f"got {tuple(topk_weights.shape)} vs {tuple(topk_ids.shape)}"
        )
    if topk_ids.device != topk_weights.device:
        raise ValueError("topk_ids and topk_weights must be on the same device")


@torch.no_grad()
def simulate_ck_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int = 32,
    expert_mask: Optional[torch.Tensor] = None,
) -> CkSortingResult:
    """Simulate AITER/CK MoE sorting metadata with torch tensors."""
    _check_topk(topk_ids, topk_weights)
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    device = topk_ids.device
    m, topk = topk_ids.shape
    topk_ids_i32 = topk_ids.to(torch.int32).contiguous()
    topk_weights_f32 = topk_weights.to(torch.float32).contiguous()

    if expert_mask is None:
        active_mask = torch.ones((num_experts,), device=device, dtype=torch.bool)
    else:
        if expert_mask.numel() != num_experts:
            raise ValueError(
                "expert_mask length must equal num_experts, "
                f"got {expert_mask.numel()} vs {num_experts}"
            )
        active_mask = expert_mask.to(device=device).to(torch.bool).flatten()

    max_num_tokens_padded = topk_ids_i32.numel() + num_experts * block_size - topk
    max_num_m_blocks = _ceil_div(max_num_tokens_padded, block_size)
    pad_id = (topk << 24) | m

    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), pad_id, dtype=torch.int32, device=device
    )
    sorted_weights = torch.zeros(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    tokens_positions_per_expert = torch.zeros(
        (num_experts * 2,), dtype=torch.int32, device=device
    )
    num_valid_ids = torch.empty((1,), dtype=torch.int32, device=device)
    m_indices = torch.full((max_num_tokens_padded,), -1, dtype=torch.int32, device=device)
    token_indices = torch.full(
        (max_num_tokens_padded,), -1, dtype=torch.int32, device=device
    )
    route_indices = torch.full(
        (max_num_tokens_padded,), -1, dtype=torch.int32, device=device
    )
    moe_c_sorted_token_ids = torch.full(
        (max_num_tokens_padded,), m * topk, dtype=torch.int32, device=device
    )
    expert_offsets = torch.zeros((num_experts + 1,), dtype=torch.int32, device=device)

    token_grid = (
        torch.arange(m, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(m, topk)
    )
    route_grid = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(m, topk)
    )

    expert_entries = []
    skipped_experts = 0
    for expert_id in range(num_experts):
        if not bool(active_mask[expert_id].item()):
            skipped_experts += 1
            expert_entries.append((expert_id, skipped_experts, None, None, None, 0))
            continue
        mask = topk_ids_i32 == expert_id
        tokens = token_grid[mask]
        routes = route_grid[mask]
        weights = topk_weights_f32[mask]
        count = int(tokens.numel())
        expert_entries.append((expert_id, skipped_experts, tokens, routes, weights, count))

    # CK sorting compacts experts that have real routes toward the front.  The
    # W8A8 decode kernels may cap EM for small M, so active blocks must appear
    # before empty expert padding blocks.
    ordered_entries = [
        entry for entry in expert_entries if entry[5] > 0 and bool(active_mask[entry[0]].item())
    ]
    ordered_entries.extend(
        entry for entry in expert_entries if entry[5] == 0 and bool(active_mask[entry[0]].item())
    )

    sorted_begin = 0
    block_begin = 0
    for expert_id, skipped_before, tokens, routes, weights, count in ordered_entries:
        num_blocks = _ceil_div(count, block_size) if count > 0 else 1
        padded_count = num_blocks * block_size
        end = sorted_begin + count
        padded_end = sorted_begin + padded_count

        if count:
            sorted_token_ids[sorted_begin:end] = (routes << 24) | tokens
            sorted_weights[sorted_begin:end] = weights
            token_indices[sorted_begin:end] = tokens
            route_indices[sorted_begin:end] = routes
            moe_c_sorted_token_ids[sorted_begin:end] = tokens * topk + routes
            m_indices[sorted_begin:end] = expert_id

        local_expert_id = expert_id - skipped_before
        sorted_expert_ids[block_begin : block_begin + num_blocks] = local_expert_id
        tokens_positions_per_expert[expert_id] = count
        tokens_positions_per_expert[num_experts + expert_id] = sorted_begin

        sorted_begin = padded_end
        block_begin += num_blocks
        expert_offsets[expert_id + 1] = sorted_begin

    last_offset = 0
    for expert_id in range(num_experts):
        if expert_offsets[expert_id + 1] == 0:
            expert_offsets[expert_id + 1] = last_offset
        else:
            last_offset = expert_offsets[expert_id + 1]

    num_valid_ids[0] = sorted_begin
    return CkSortingResult(
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        tokens_positions_per_expert=tokens_positions_per_expert,
        m_indices=m_indices,
        token_indices=token_indices,
        route_indices=route_indices,
        moe_c_sorted_token_ids=moe_c_sorted_token_ids,
        expert_offsets=expert_offsets,
        active_expert_mask=active_mask,
    )


@torch.no_grad()
def gather_by_ck_sorting(
    x: torch.Tensor,
    sorting: CkSortingResult,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Gather token rows according to simulate_ck_moe_sorting order."""
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension")
    valid = sorting.token_indices >= 0
    out = x.new_full((sorting.token_indices.numel(), *x.shape[1:]), fill_value)
    if bool(valid.any().item()):
        out[valid] = x[sorting.token_indices[valid].to(torch.long)]
    return out
