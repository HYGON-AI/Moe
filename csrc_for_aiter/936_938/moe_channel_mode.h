// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

// HIPC S F K MM VVV: stage, sorting, family, BM / 16, profile.
constexpr int encode_moe_channel_mode(int stage, int sorting, int family, int bm, int profile)
{
    return stage * 10000000 + sorting * 1000000 + family * 100000 + (bm / 16) * 1000 + profile;
}

constexpr bool is_moe_channel_mode(int64_t mode) {
  return mode >= 10000000LL && mode < 30000000LL;
}

// Kept as a named alias for callers that still describe the FP8 path.
constexpr bool is_fp8_channel_mode(int64_t mode) {
  return is_moe_channel_mode(mode);
}

constexpr int moe_channel_mode_stage(int64_t mode) {
  return static_cast<int>(mode / 10000000LL);
}

constexpr int moe_channel_mode_sorting(int64_t mode) {
  return static_cast<int>((mode / 1000000LL) % 10);
}

constexpr int moe_channel_mode_family(int64_t mode) {
  return static_cast<int>((mode / 100000LL) % 10);
}

constexpr int moe_channel_mode_bm(int64_t mode) {
  return static_cast<int>((mode / 1000LL) % 100) * 16;
}

constexpr int moe_channel_mode_profile(int64_t mode) {
    return static_cast<int>(mode % 1000LL);
}
