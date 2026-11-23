# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared HIPC mode encoding: stage, sorting, family, BM / 16, profile."""


def mode(stage: int, sorting: int, family: int, bm: int, profile: int) -> int:
    if stage not in (1, 2) or sorting not in (0, 1) or not 0 <= family < 10:
        raise ValueError("invalid mode prefix")
    if bm <= 0 or bm % 16 or bm // 16 >= 100:
        raise ValueError(f"BM is not representable: {bm}")
    if not 0 < profile < 1000:
        raise ValueError(f"profile is not representable: {profile}")
    return stage * 10_000_000 + sorting * 1_000_000 + family * 100_000 + (bm // 16) * 1_000 + profile
