# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Select the same moe_c backend the containing checkout uses."""

import importlib
from pathlib import Path
import sys


MOE_ROOT = Path(__file__).resolve().parents[1]
AITER_ROOT = MOE_ROOT.parents[1]
IN_AITER = (AITER_ROOT / "aiter/jit/optCompilerConfig.json").is_file()
CHECKOUT_ROOT = AITER_ROOT if IN_AITER else MOE_ROOT
CONFIG_ROOT = AITER_ROOT / "aiter/moe_c_configs" if IN_AITER else MOE_ROOT / "moe_c_configs"

if IN_AITER:
    sys.path.insert(0, str(AITER_ROOT))


def fused_moe():
    return importlib.import_module("aiter.fused_moe_c" if IN_AITER else "moe_utils.fused_moe_c")


def ops():
    return importlib.import_module("aiter" if IN_AITER else "moe_utils.moe_ops")
