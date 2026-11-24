// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// W4A16 MoE prefill GEMM (gemm1 = gate/up, gemm2 = down) for large M.
//
// Layout (same tensors as the decode kernels in moe_w4a16_opt.h):
//   input   [rows, size_k]                 bf16/fp16, rows gathered through sorted_token_ids
//   qweight [E, size_n, size_k / 8]        uint32, 8 int4 along K per word (low nibble first)
//   scale   [E, size_n, size_k / GS]       group GS = 32 or 128 along K, w4a16_marlin_scale shuffled
//   output  [size_m * top_k, size_n]       row = sorted token id
// Symmetric quantization only (zero point fixed at 8, qzeros ignored), like the decode kernels.
//
// Tiling: every warp owns all BM rows of the block and NN * 16 output columns, so a dequantized
// B fragment is reused by all BM / 16 m-tiles (dequant VALU does not overlap with MMAC on gfx936,
// so this reuse is what bounds the kernel).  A goes through an LDS tile of BM x 128 per K step
// (double-buffered: one barrier per step, or single-buffered: two barriers, half the LDS); B is
// loaded coalesced one step ahead and transposed to the MMAC layout with ds_bpermute.  With
// size_k == 256 (gemm2) the whole A tile stays in LDS and a block walks NLoop column chunks.
//
// MMAC 16x16x16 operands: lane l supplies row l % 16 and k 4 * (l / 16) + [0, 4); the result
// register r of lane l is C[l % 16][4 * r + l / 16].  Inside one 128-wide K step lane group
// c = l / 16 covers the contiguous k range [32 c, 32 c + 32) for both A and B, i.e. scale group
// c of the step with GS = 32 and the single group of the step with GS = 128.
// B row j of n-tile t is the output column 4 * NN * (j % 4) + 4 * t + j / 4 of the warp, which
// makes the 4 * NN accumulator values of a lane contiguous output columns.

#pragma once

#include <algorithm>
#include <hip/hip_runtime.h>
#include <type_traits>

#include "moe_w4a16_prefill_pipeline.h"

namespace w4a16_prefill
{

using f32x2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using i32x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using i32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using u32x2 = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;

constexpr int BK = 128; // K per pipeline step

// Scale group (inside its K step) of lane group lk: the shuffled scale of group j of step s is at
// j * steps + s (w4a16_marlin_scale), so a lane group reads the scales of consecutive steps.
template <int GS>
__device__ __forceinline__ int scale_lane_group(int lk)
{
  static_assert(GS == 32 || GS == 128, "W4A16 group size along K");
  return GS == 32 ? lk : 0;
}

// Barrier that only waits for LDS traffic.  __syncthreads() also drains vmcnt, which would
// serialize the global prefetch issued right before it.
__device__ __forceinline__ void lds_barrier()
{
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void *ptr)
{
  return __builtin_hcu_make_buffer_rsrc(const_cast<void *>(ptr), (short)0, 0x7fffffff, 0x00020000);
}

template <typename T>
__device__ __forceinline__ f32x4 mmac(i32x2 a, i32x2 b, f32x4 c)
{
  if constexpr (std::is_same_v<T, __hip_bfloat16>)
  {
    return __builtin_hcu_mmac_f32_16x16x16_bf16(__builtin_bit_cast(vec4_bf16, a), __builtin_bit_cast(vec4_bf16, b), c);
  }
  else
  {
    return __builtin_hcu_mmac_f32_16x16x16_f16(__builtin_bit_cast(vec4_fp16, a), __builtin_bit_cast(vec4_fp16, b), c);
  }
}

template <int BYTE>
__device__ __forceinline__ float cvt_ubyte(uint32_t x)
{
  float r;
  if constexpr (BYTE == 0)
    asm("v_cvt_f32_ubyte0 %0, %1" : "=v"(r) : "v"(x));
  else if constexpr (BYTE == 1)
    asm("v_cvt_f32_ubyte1 %0, %1" : "=v"(r) : "v"(x));
  else if constexpr (BYTE == 2)
    asm("v_cvt_f32_ubyte2 %0, %1" : "=v"(r) : "v"(x));
  else
    asm("v_cvt_f32_ubyte3 %0, %1" : "=v"(r) : "v"(x));
  return r;
}

// Per n-tile dequant constants of one K step.
// bf16: mul = {s, s / 16} (the odd nibbles stay in the high half of each byte), add = -8 s.
// fp16: h2 = packed {s, s}.
struct DeqScale
{
  f32x2 mul;
  f32x2 add;
  uint32_t h2;
};

template <typename T>
__device__ __forceinline__ DeqScale make_scale(uint32_t sbits)
{
  DeqScale d;
  if constexpr (std::is_same_v<T, __hip_bfloat16>)
  {
    const float s = __builtin_bit_cast(float, sbits << 16);
    d.mul = f32x2{s, s * 0.0625f};
    d.add = f32x2{-8.f * s, -8.f * s};
    d.h2 = 0;
  }
  else
  {
    d.mul = f32x2{0.f, 0.f};
    d.add = f32x2{0.f, 0.f};
    d.h2 = sbits | (sbits << 16);
  }
  return d;
}

// 8 int4 (k0..k7, low nibble first) -> 8 values (q - 8) * scale packed as 4 x 2 x 16 bit.
// bf16: natural k order, fp32 math and truncation (same numerics as dequant_gfx936).
// fp16: packed half math, k order (0,4),(1,5),(2,6),(3,7); A is permuted to match in LDS.
template <typename T>
__device__ __forceinline__ i32x4 dequant8(uint32_t q, const DeqScale &d)
{
  i32x4 r;
  if constexpr (std::is_same_v<T, __hip_bfloat16>)
  {
    const uint32_t lo = q & 0x0f0f0f0fu; // k0, k2, k4, k6
    const uint32_t hi = q & 0xf0f0f0f0u; // 16 * (k1, k3, k5, k7)
    f32x2 p0 = {cvt_ubyte<0>(lo), cvt_ubyte<0>(hi)};
    f32x2 p1 = {cvt_ubyte<1>(lo), cvt_ubyte<1>(hi)};
    f32x2 p2 = {cvt_ubyte<2>(lo), cvt_ubyte<2>(hi)};
    f32x2 p3 = {cvt_ubyte<3>(lo), cvt_ubyte<3>(hi)};
    p0 = __builtin_elementwise_fma(p0, d.mul, d.add);
    p1 = __builtin_elementwise_fma(p1, d.mul, d.add);
    p2 = __builtin_elementwise_fma(p2, d.mul, d.add);
    p3 = __builtin_elementwise_fma(p3, d.mul, d.add);
    r[0] = (int)__builtin_amdgcn_perm(__builtin_bit_cast(u32x2, p0)[1], __builtin_bit_cast(u32x2, p0)[0], 0x07060302u);
    r[1] = (int)__builtin_amdgcn_perm(__builtin_bit_cast(u32x2, p1)[1], __builtin_bit_cast(u32x2, p1)[0], 0x07060302u);
    r[2] = (int)__builtin_amdgcn_perm(__builtin_bit_cast(u32x2, p2)[1], __builtin_bit_cast(u32x2, p2)[0], 0x07060302u);
    r[3] = (int)__builtin_amdgcn_perm(__builtin_bit_cast(u32x2, p3)[1], __builtin_bit_cast(u32x2, p3)[0], 0x07060302u);
  }
  else
  {
    using h2 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
    const h2 s = __builtin_bit_cast(h2, d.h2);
    const uint32_t q1 = q >> 8;
    const uint32_t lo0 = (q & 0x000f000fu) | 0x64006400u;  // 1024 + k0, 1024 + k4
    const uint32_t hi0 = (q & 0x00f000f0u) | 0x64006400u;  // 1024 + 16 k1, ...
    const uint32_t lo1 = (q1 & 0x000f000fu) | 0x64006400u; // 1024 + k2, 1024 + k6
    const uint32_t hi1 = (q1 & 0x00f000f0u) | 0x64006400u;
    const h2 sub = {(_Float16)-1032.f, (_Float16)-1032.f};
    const h2 mul = {(_Float16)0.0625f, (_Float16)0.0625f};
    const h2 add = {(_Float16)-72.f, (_Float16)-72.f};
    r[0] = __builtin_bit_cast(int, (__builtin_bit_cast(h2, lo0) + sub) * s);
    r[1] = __builtin_bit_cast(int, __builtin_elementwise_fma(__builtin_bit_cast(h2, hi0), mul, add) * s);
    r[2] = __builtin_bit_cast(int, (__builtin_bit_cast(h2, lo1) + sub) * s);
    r[3] = __builtin_bit_cast(int, __builtin_elementwise_fma(__builtin_bit_cast(h2, hi1), mul, add) * s);
  }
  return r;
}

// two fp32 -> packed 16 bit, round to nearest even
template <typename T>
__device__ __forceinline__ uint32_t pack2(float x, float y)
{
  if constexpr (std::is_same_v<T, __hip_bfloat16>)
  {
    uint32_t a = __builtin_bit_cast(uint32_t, x);
    uint32_t b = __builtin_bit_cast(uint32_t, y);
    a += 0x7fffu + ((a >> 16) & 1u);
    b += 0x7fffu + ((b >> 16) & 1u);
    return __builtin_amdgcn_perm(b, a, 0x07060302u);
  }
  else
  {
    using h2 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
    h2 h = {(_Float16)x, (_Float16)y};
    return __builtin_bit_cast(uint32_t, h);
  }
}

// OPT bits of the prefill kernel (0 = the original numerics and schedule):
//   OPT_FRESH: with NLoop > 1 the first MMAC of every column chunk reads a zero quad held in
//              registers instead of the accumulators, so the 4 * NM * NN v_mov of zero_acc()
//              between chunks disappear (MMAC allows dst != src2; an inline 0 is not encodable).
//   OPT_RHU:   bf16 output of the non-SILU epilogue rounds half up in magnitude (x + 0x8000)
//              instead of to nearest even; the two differ only on exact ties (low 16 bits ==
//              0x8000) and save one VALU per value (1-4% of gemm2).
//   OPT_SWZ:   with 16 chunks per K step (AH = 1) the LDS row slot of A chunk c is r ^ ((c >> 1) & 7)
//              instead of r ^ (c & 7).  The fragment read c = 4 lk + u is 2-way bank conflicted
//              with the old slot (16.2 instead of 8.2 cycles per ds_read_b128 on gfx936, ~30% of the
//              gemm2 cycles); the new one is conflict free for the reads at 16 instead of 10 cycles
//              per A store, which are 50x rarer.  8 chunks per step (AH = 2) keep r ^ (c & 7), which
//              is conflict free there.  Only the LDS placement changes, not the numerics.
constexpr int OPT_FRESH = 1;
constexpr int OPT_RHU = 2;
constexpr int OPT_SWZ = 4;

template <int OPT, int CHUNKS>
__device__ __forceinline__ int a_swz(int c)
{
  if constexpr ((OPT & OPT_SWZ) != 0 && CHUNKS == 16)
    return (c >> 1) & 7;
  else
    return c & 7;
}

// two fp32 -> packed bf16, round half up in magnitude (ties away from zero)
__device__ __forceinline__ uint32_t pack2_bf16_rhu(float x, float y)
{
  const uint32_t a = __builtin_bit_cast(uint32_t, x) + 0x8000u;
  const uint32_t b = __builtin_bit_cast(uint32_t, y) + 0x8000u;
  return __builtin_amdgcn_perm(b, a, 0x07060302u);
}

// round to 16 bit (nearest even) and back
template <typename T>
__device__ __forceinline__ float round16(float x)
{
  if constexpr (std::is_same_v<T, __hip_bfloat16>)
  {
    uint32_t a = __builtin_bit_cast(uint32_t, x);
    a += 0x7fffu + ((a >> 16) & 1u);
    return __builtin_bit_cast(float, a & 0xffff0000u);
  }
  else
  {
    return (float)(_Float16)x;
  }
}

// silu(gate) * up of the fp32 gemm1 accumulators, rounded once at the store.  The unfused
// gemm1 + silu_and_mul path also rounds gate, up and silu(gate) to 16 bit; define
// MOE_W4A16_EXACT_SILU to reproduce it bit for bit (2.5% slower gemm1).
template <typename T>
__device__ __forceinline__ float silu_mul(float gate, float up)
{
#ifdef MOE_W4A16_EXACT_SILU
  const float g = round16<T>(gate);
  float a = g * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(-g * 1.44269504088896340736f));
  if constexpr (std::is_same_v<T, __hip_bfloat16>)
    a = round16<T>(a);
  return a * round16<T>(up);
#else
  return gate * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(-gate * 1.44269504088896340736f)) * up;
#endif
}

// two outputs at once: the multiplies / adds go to packed fp32 VALU ops (exp2 / rcp stay scalar)
template <typename T>
__device__ __forceinline__ f32x2 silu_mul2(f32x2 gate, f32x2 up)
{
#ifdef MOE_W4A16_EXACT_SILU
  return f32x2{silu_mul<T>(gate[0], up[0]), silu_mul<T>(gate[1], up[1])};
#else
  const f32x2 e = gate * f32x2{-1.44269504088896340736f, -1.44269504088896340736f};
  const f32x2 d = f32x2{__builtin_amdgcn_exp2f(e[0]), __builtin_amdgcn_exp2f(e[1])} + f32x2{1.0f, 1.0f};
  return gate * f32x2{__builtin_amdgcn_rcpf(d[0]), __builtin_amdgcn_rcpf(d[1])} * up;
#endif
}

template <typename T, int BM, int BN, int NLoop, W4A16Pipeline Pipeline, W4A16Epilogue Epilogue, int GS>
__global__ void __launch_bounds__((Tile<BM, BN, Pipeline>::WARPS * 64))
    __attribute__((amdgpu_waves_per_eu(Tile<BM, BN, Pipeline>::MinWaves, 8))) moe_w4a16_prefill_kernel(
    const T *__restrict__ input,
    const uint32_t *__restrict__ qweight,
    T *__restrict__ output,
    const T *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    int size_m,
    int size_n,
    int size_k,
    int sorted_token_lens,
    int top_k)
{
  using Config = Tile<BM, BN, Pipeline>;
  constexpr int WARPS = Config::WARPS, NN = Config::NN, ST = Config::ST;
  constexpr int SCM = Config::SCM, AH = Config::AH, OPT = Config::OPT;
  constexpr bool PREFETCH_A = Config::PREFETCH_A;
  constexpr bool MUL_TOPK = Epilogue == W4A16Epilogue::DownWeighted, SILU = Epilogue == W4A16Epilogue::UpSilu;
  constexpr int NT = WARPS * 64;
  constexpr int NM = BM / 16;
  constexpr int ROW_CHUNKS = BK * (int)sizeof(T) / 16; // 16-byte chunks per A row and K step
  // AH = 2: A goes through LDS in two half steps per K step (half the LDS per stage).  Half h
  // holds, for every lane group lk, the k range [32 lk + 16 h, 32 lk + 16 h + 16), i.e. sub-steps
  // u = 2 h, 2 h + 1 of the compute loop, so B, the scales and the dequant stay per 128-K step.
  constexpr int CHUNKS = ROW_CHUNKS / AH; // A chunks per row in one LDS stage
  constexpr int STAGE_CHUNKS = BM * CHUNKS;
  constexpr int A_LOADS = (STAGE_CHUNKS + NT - 1) / NT;
  static_assert(BM % 16 == 0 && ROW_CHUNKS == 16 && NT % 16 == 0, "tile shape");
  static_assert(AH == 1 || (AH == 2 && ST == 2 && NLoop == 1), "half-step A staging is a streamed ST2 pipeline");
  const int m_block = blockIdx.y;
  const int num_valid = size_m * top_k;
  if (m_block * BM >= num_tokens_post_pad[0])
    return;
  if (sorted_token_ids[m_block * BM] >= num_valid)
    return;

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = __builtin_hcu_readfirstlane(tid >> 6);
  const int lrow = lane & 15; // MMAC row / column inside a 16x16 tile
  const int lk = lane >> 4;   // MMAC k lane group
  const int expert = expert_ids[m_block];
  const int m_base = m_block * BM;
  const int steps = size_k / BK;

  // LDS A stage: [m-tile][16 byte chunk c of the K step][row slot], row r of the m-tile stored at
  // slot r ^ (c & 7).  A fragment read (16 lanes = 16 rows of one chunk) is a permutation of one
  // 256 B block and a row-contiguous store (16 lanes = 16 chunks of one row) hits 8 distinct
  // banks per 8 lanes, so both directions are conflict free (row-major tiles are 2-way conflicted
  // on the fragment read for any XOR swizzle).
  extern __shared__ i32x4 lds_a[];

  // ---- A gather: slot i of thread tid = 16 byte chunk (i * NT + tid) of the row-major tile ---
  const __amdgpu_buffer_rsrc_t rsrc_a = make_rsrc(input);
  int a_off[A_LOADS];
  int a_lds[A_LOADS];
#pragma unroll
  for (int i = 0; i < A_LOADS; i++)
  {
    const int slot = min(i * NT + tid, STAGE_CHUNKS - 1);
    const int row = slot / CHUNKS;
    const int chunk = slot % CHUNKS;                                  // chunk in the stage
    const int gchunk = AH == 1 ? chunk : 4 * (chunk >> 1) + (chunk & 1); // chunk of the K step (half 0)
    const int tok = sorted_token_ids[min(m_base + row, sorted_token_lens - 1)];
    const int src = tok < num_valid ? tok / top_k : 0;
    a_off[i] = src * size_k * (int)sizeof(T) + gchunk * 16;
    a_lds[i] = ((row / 16) * CHUNKS + chunk) * 16 + ((row & 15) ^ a_swz<OPT, CHUNKS>(chunk));
  }

  // ---- B / scale setup ----------------------------------------------------------------------
  // B row j of n-tile t is output column n_warp + 4 NN (j % 4) + 4 t + j / 4.  B is loaded with 4
  // adjacent lanes per row (64 contiguous bytes, lane 4 j + c holds chunk c of row j) and then
  // transposed with ds_bpermute to the MMAC layout (lane j + 16 c).
  const int groups = size_k / GS;
  const int lgrp = scale_lane_group<GS>(lk);
  const __amdgpu_buffer_rsrc_t rsrc_b = make_rsrc(qweight + (size_t)expert * size_n * (size_k / 8));
  const __amdgpu_buffer_rsrc_t rsrc_s = make_rsrc(weight_scale + (size_t)expert * size_n * groups);
  const int b_row_bytes = size_k / 2;
  const int lq = lane >> 2;
  const int b_perm_addr = (4 * lrow + lk) * 4;
  // SILU (gemm1 + silu_and_mul): tiles t < NN / 2 are gate columns and tile t + NN / 2 the up
  // columns (B rows + size_n / 2) of the same outputs, so a lane owns both halves of its outputs.
  // n_warp counts output columns.
  constexpr int NH = SILU ? NN / 2 : NN;
  const int up_rows = SILU ? size_n / 2 : 0;
  int n_warp = (blockIdx.x * NLoop) * (BN / (NN / NH)) + warp * 4 * 4 * NH;
  // B row of row j of n-tile t
  auto wcol = [&](int t, int j) {
    return n_warp + 4 * NH * (j & 3) + 4 * (t % NH) + (j >> 2) + (t >= NH ? up_rows : 0);
  };
  auto b_offset = [&](int t, int step) {
    return wcol(t, lq) * b_row_bytes + (lane & 3) * 16 + step * (BK / 2);
  };
  // scale of the lane's group lgrp of K step `step` (see scale_lane_group)
  auto s_offset = [&](int t, int step) {
    return (wcol(t, lrow) * groups + lgrp * steps + step) * (int)sizeof(T);
  };

  // q = K step (AH == 1) or half step (AH == 2)
  auto load_a = [&](i32x4(&regs)[A_LOADS], int q) {
    const int off = (q / AH) * BK * (int)sizeof(T) + (q % AH) * 32;
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
      if (A_LOADS * NT == STAGE_CHUNKS || i * NT + tid < STAGE_CHUNKS)
        regs[i] = __builtin_hcu_raw_buffer_load_b128(rsrc_a, a_off[i] + off, 0, 0);
  };
  auto store_a = [&](const i32x4(&regs)[A_LOADS], int stage) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
    {
      if (A_LOADS * NT == STAGE_CHUNKS || i * NT + tid < STAGE_CHUNKS)
      {
        i32x4 v = regs[i];
        if constexpr (!std::is_same_v<T, __hip_bfloat16>)
        {
          // k order (0,4),(1,5),(2,6),(3,7) to match the fp16 dequant
          const uint32_t x0 = v[0], x1 = v[1], x2 = v[2], x3 = v[3];
          v[0] = (int)__builtin_amdgcn_perm(x2, x0, 0x05040100u);
          v[1] = (int)__builtin_amdgcn_perm(x2, x0, 0x07060302u);
          v[2] = (int)__builtin_amdgcn_perm(x3, x1, 0x05040100u);
          v[3] = (int)__builtin_amdgcn_perm(x3, x1, 0x07060302u);
        }
        lds_a[stage * STAGE_CHUNKS + a_lds[i]] = v;
      }
    }
  };
  auto load_b = [&](i32x4(&regs)[NN], int step) {
#pragma unroll
    for (int t = 0; t < NN; t++)
      regs[t] = __builtin_hcu_raw_buffer_load_b128(rsrc_b, b_offset(t, step), 0, 0);
  };
  auto permute_b = [&](i32x4(&regs)[NN]) {
#pragma unroll
    for (int t = 0; t < NN; t++)
#pragma unroll
      for (int d = 0; d < 4; d++)
        regs[t][d] = __builtin_amdgcn_ds_bpermute(b_perm_addr, regs[t][d]);
  };
  // SC consecutive step scales (2 or 4) per n-tile
  auto load_s = [&](auto sc_tag, u32x2(&regs)[NN], int step) {
    constexpr int SC = decltype(sc_tag)::value;
#pragma unroll
    for (int t = 0; t < NN; t++)
    {
      if constexpr (SC == 4)
        regs[t] = __builtin_bit_cast(u32x2, __builtin_hcu_raw_buffer_load_b64(rsrc_s, s_offset(t, step), 0, 0));
      else
        regs[t][0] = (uint32_t)__builtin_hcu_raw_buffer_load_b32(rsrc_s, s_offset(t, step), 0, 0);
    }
  };

  f32x4 acc[NM][NN];
  auto zero_acc = [&]() {
#pragma unroll
    for (int mt = 0; mt < NM; mt++)
#pragma unroll
      for (int t = 0; t < NN; t++)
        acc[mt][t] = f32x4{0.f, 0.f, 0.f, 0.f};
  };

  // one 128-wide K step (half h of it for AH == 2) from LDS stage `stage`, B (MMAC layout) and
  // the scales of sub-step k
  // zero quad for OPT_FRESH, opaque so that the compiler keeps it in registers
  f32x4 zq = {0.f, 0.f, 0.f, 0.f};
  if constexpr ((OPT & OPT_FRESH) != 0)
    asm volatile("" : "+v"(zq));
  auto compute = [&](int stage, const i32x4(&breg)[NN], const u32x2(&sreg)[NN], int k, int h = 0, bool fresh = false) {
    DeqScale ds[NN];
#pragma unroll
    for (int t = 0; t < NN; t++)
      ds[t] = make_scale<T>((sreg[t][k >> 1] >> ((k & 1) * 16)) & 0xffffu);
    const i32x4 *a_stage = lds_a + stage * STAGE_CHUNKS;
    constexpr int U0 = 0, U1 = 4 / AH; // sub-steps of this call: u = U0 + 2 h .. U1 + 2 h
    auto load_frag = [&](i32x4(&a)[NM], int u) {
      const int c = AH == 1 ? 4 * lk + u : 2 * lk + (u & 1);
#pragma unroll
      for (int mt = 0; mt < NM; mt++)
        a[mt] = a_stage[(mt * CHUNKS + c) * 16 + (lrow ^ a_swz<OPT, CHUNKS>(c))];
    };
    i32x4 a_cur[NM], a_nxt[NM];
    load_frag(a_cur, U0 + 2 * h);
#pragma unroll
    for (int uu = U0; uu < U1; uu++)
    {
      const int u = uu + 2 * h;
      if (PREFETCH_A && uu < U1 - 1)
        load_frag(a_nxt, u + 1);
#pragma unroll
      for (int t = 0; t < NN; t++)
      {
        const i32x4 b = dequant8<T>((uint32_t)breg[t][u], ds[t]);
        const i32x2 b0 = {b[0], b[1]}, b1 = {b[2], b[3]};
#pragma unroll
        for (int mt = 0; mt < NM; mt++)
        {
          const i32x2 a0 = {a_cur[mt][0], a_cur[mt][1]}, a1 = {a_cur[mt][2], a_cur[mt][3]};
          acc[mt][t] = mmac<T>(a0, b0, (fresh && uu == U0) ? zq : acc[mt][t]);
          acc[mt][t] = mmac<T>(a1, b1, acc[mt][t]);
        }
      }
      if (uu < U1 - 1)
      {
        if constexpr (PREFETCH_A)
        {
#pragma unroll
          for (int mt = 0; mt < NM; mt++)
            a_cur[mt] = a_nxt[mt];
        }
        else
        {
          load_frag(a_cur, u + 1);
        }
      }
    }
  };

  // Token / topk weight of the lane's rows, loaded once up front: loads issued in the epilogue
  // queue behind the outstanding B / A prefetch on vmcnt (5-14% of gemm2 with its NLoop epilogues).
  int row_tok[NM];
  float row_w[MUL_TOPK ? NM : 1];
  auto load_rows = [&]() {
#pragma unroll
    for (int mt = 0; mt < NM; mt++)
      row_tok[mt] = sorted_token_ids[min(m_base + mt * 16 + lrow, sorted_token_lens - 1)];
    if constexpr (MUL_TOPK)
    {
#pragma unroll
      for (int mt = 0; mt < NM; mt++)
        row_w[mt] = row_tok[mt] < num_valid ? topk_weights[row_tok[mt]] : 1.f;
    }
  };

  // lane holds rows mt * 16 + lrow and 4 * NH contiguous output columns
  auto epilogue = [&]() {
    const int col = n_warp + 4 * NH * lk;
#pragma unroll
    for (int mt = 0; mt < NM; mt++)
    {
      const int tok = row_tok[mt];
      if constexpr (SILU)
      {
        if (tok < num_valid)
        {
          uint32_t packed[2 * NH];
#pragma unroll
          for (int t = 0; t < NH; t++)
          {
            const f32x4 g = acc[mt][t], u = acc[mt][t + NH];
            const f32x2 lo = silu_mul2<T>(f32x2{g[0], g[1]}, f32x2{u[0], u[1]});
            const f32x2 hi = silu_mul2<T>(f32x2{g[2], g[3]}, f32x2{u[2], u[3]});
            packed[2 * t] = pack2<T>(lo[0], lo[1]);
            packed[2 * t + 1] = pack2<T>(hi[0], hi[1]);
          }
          // nontemporal: the activation is streamed once by gemm2 (~1-2% faster than plain stores)
          T *dst = output + (size_t)tok * (size_n / 2) + col;
          if constexpr (NH % 2 == 0)
          {
#pragma unroll
            for (int v = 0; v < NH / 2; v++)
              __builtin_nontemporal_store(
                  i32x4{(int)packed[4 * v], (int)packed[4 * v + 1], (int)packed[4 * v + 2], (int)packed[4 * v + 3]},
                  reinterpret_cast<i32x4 *>(dst) + v);
          }
          else
          {
            __builtin_nontemporal_store(i32x2{(int)packed[0], (int)packed[1]}, reinterpret_cast<i32x2 *>(dst));
          }
        }
      }
      else if (tok < num_valid)
      {
        float w = 1.f;
        if constexpr (MUL_TOPK)
          w = row_w[mt];
        uint32_t packed[2 * NN];
#pragma unroll
        for (int t = 0; t < NN; t++)
        {
          const f32x2 w2 = {w, w}; // packed multiplies
          const f32x2 lo = f32x2{acc[mt][t][0], acc[mt][t][1]} * w2;
          const f32x2 hi = f32x2{acc[mt][t][2], acc[mt][t][3]} * w2;
          if constexpr ((OPT & OPT_RHU) != 0 && std::is_same_v<T, __hip_bfloat16>)
          {
            packed[2 * t] = pack2_bf16_rhu(lo[0], lo[1]);
            packed[2 * t + 1] = pack2_bf16_rhu(hi[0], hi[1]);
          }
          else
          {
            packed[2 * t] = pack2<T>(lo[0], lo[1]);
            packed[2 * t + 1] = pack2<T>(hi[0], hi[1]);
          }
        }
        i32x4 *dst = reinterpret_cast<i32x4 *>(output + (size_t)tok * size_n + col);
#pragma unroll
        for (int v = 0; v < NN / 2; v++)
          // nontemporal: streamed once by the next kernel (1-3.5% faster gemm2 with the row cache)
          __builtin_nontemporal_store(
              i32x4{(int)packed[4 * v], (int)packed[4 * v + 1], (int)packed[4 * v + 2], (int)packed[4 * v + 3]}, dst + v);
      }
    }
  };

  i32x4 areg[A_LOADS];
  i32x4 b0[NN], b1[NN];
  u32x2 sc[NN], sc_next[NN];
  if (NLoop == 1 || (OPT & OPT_FRESH) == 0)
    zero_acc();

  if constexpr (NLoop > 1)
  {
    // ---- resident A (size_k == 2 * BK): A loaded once, NLoop column chunks of BN per block ----
    std::integral_constant<int, 2> sc2;
    load_rows();
    load_a(areg, 0);
    load_b(b0, 0);
    load_s(sc2, sc, 0);
    load_b(b1, 1);
    store_a(areg, 0);
    load_a(areg, 1);
    permute_b(b0);
    store_a(areg, 1);
    lds_barrier();
    constexpr bool FRESH = (OPT & OPT_FRESH) != 0;
    for (int ch = 0; ch < NLoop; ch++)
    {
      const bool more = ch + 1 < NLoop;
      compute(0, b0, sc, 0, 0, FRESH);
      permute_b(b1);
      if (more)
      {
        n_warp += BN;
        load_b(b0, 0);
        load_s(sc2, sc_next, 0);
        n_warp -= BN;
      }
      compute(1, b1, sc, 1);
      epilogue();
      if (more)
      {
        n_warp += BN;
        load_b(b1, 1);
        permute_b(b0);
#pragma unroll
        for (int t = 0; t < NN; t++)
          sc[t] = sc_next[t];
        if constexpr (!FRESH)
          zero_acc();
      }
    }
    return;
  }
  else
  {
    // ---- streamed A: A(s + 1) / B(s + 1) in flight while step s computes --------------------
    auto pipeline = [&](auto sc_tag) {
      constexpr int SC = decltype(sc_tag)::value;
      const int halves = steps * AH;
      load_rows();
      load_a(areg, 0);
      load_b(b0, 0);
      load_s(sc_tag, sc, 0);
      load_b(b1, 1);
      store_a(areg, 0);
      permute_b(b0);
      load_a(areg, 1);
      lds_barrier();
      // Loads are issued unconditionally (step clamped to the last one): data-dependent branches
      // around them make the waitcnt pass fall back to vmcnt(0), which drains the prefetch.
      for (int s = 0; s < steps; s += SC)
      {
        load_s(sc_tag, sc_next, min(s + SC, steps - SC));
#pragma unroll
        for (int k = 0; k < SC; k++)
        {
          const int st = s + k;
          i32x4(&bc)[NN] = (k & 1) ? b1 : b0;
          i32x4(&bn)[NN] = (k & 1) ? b0 : b1;
          if constexpr (AH == 2)
          {
            // half 0 of step st from stage 0, half 1 from stage 1 (half steps alternate stages)
            compute(0, bc, sc, k, 0);
            store_a(areg, 1);
            load_a(areg, min(2 * st + 2, halves - 1));
            lds_barrier();
            compute(1, bc, sc, k, 1);
            store_a(areg, 0);
            permute_b(bn);
            load_a(areg, min(2 * st + 3, halves - 1));
            load_b(bc, min(st + 2, steps - 1));
            lds_barrier();
            continue;
          }
          if constexpr (ST == 2)
          {
            compute(k & 1, bc, sc, k);
            store_a(areg, (k + 1) & 1);
          }
          else
          {
            // single LDS stage (half the LDS, two blocks per CU): wait for every warp to finish
            // reading A(st) before overwriting it with A(st + 1)
            compute(0, bc, sc, k);
            lds_barrier();
            store_a(areg, 0);
          }
          permute_b(bn);
          load_a(areg, min(st + 2, steps - 1));
          load_b(bc, min(st + 2, steps - 1));
          lds_barrier();
        }
#pragma unroll
        for (int t = 0; t < NN; t++)
          sc[t] = sc_next[t];
      }
    };
    if (SCM >= 4 && steps % 4 == 0)
      pipeline(std::integral_constant<int, 4>{});
    else
      pipeline(std::integral_constant<int, 2>{});
    epilogue();
  }
}

// ---- gemm2 with the dequantized B resident in registers (size_k == 256) ---------------------
// A CTA owns BN = WARPS * 32 output columns and walks `run_blocks` consecutive routing blocks of TM
// rows.  Each warp dequantizes its 32 columns over the whole K = 256 once per expert (2 * 2 * 4
// MMAC fragments = 64 VGPRs) and streams the A tiles, double-buffered in LDS, through them, so the
// dequant VALU is paid once per expert run instead of once per BM rows (the largest VALU cost of
// the resident-A kernel above).  The B and scales of the next expert are loaded one tile ahead, A
// rows and topk weights two tiles ahead (see the pipeline below); one LDS barrier per tile.  The
// token row and topk weight of each tile row are fetched once per CTA, by the thread that loads the
// row's first A chunk, and published in LDS instead of being loaded by all warps.  Before the store
// the 16 byte output chunks are transposed with ds_bpermute so that 4 adjacent lanes write the 64
// contiguous bytes of one row.  Output rows are scaled by topk_weights like the other gemm2
// kernels; the bf16 output rounding (OPT_RHU) and the fp32 accumulation order match mode 525 bit
// for bit.  16 warps x ~180 VGPRs: one CTA (4 waves per SIMD) per CU.  As mode 526 it is 12-16%
// faster than mode 525 for M >= 8192 on gfx936, on par at M = 4096 and slower below (short expert
// runs).
constexpr int BstatNN = 2; // one 16-byte output chunk per lane and m-tile
constexpr int BstatWaveColumns = BstatNN * 16;
constexpr int BstatMinWaves = 4;
constexpr int BstatOpt = OPT_RHU | OPT_SWZ;

template <typename T, int TM, int BN, int GS>
__global__ void __launch_bounds__(BN / BstatWaveColumns * 64)
    __attribute__((amdgpu_waves_per_eu(BstatMinWaves, 8))) moe_w4a16_prefill_g2_bstat_kernel(
    const T *__restrict__ input,
    const uint32_t *__restrict__ qweight,
    T *__restrict__ output,
    const T *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    int size_m,
    int size_n,
    int size_k,
    int top_k,
    int run_blocks)
{
  constexpr int WARPS = BN / BstatWaveColumns, NN = BstatNN, OPT = BstatOpt;
  constexpr int NT = WARPS * 64;
  constexpr int NM = TM / 16;
  constexpr int KS = 2;                                // K steps of BK (size_k == 256)
  constexpr int CH = BK * (int)sizeof(T) / 16;         // 16-byte chunks per row and K step
  constexpr int STEP_CHUNKS = TM * CH;                 // one K step of one tile
  constexpr int TILE_CHUNKS = KS * STEP_CHUNKS;        // one LDS buffer
  constexpr int A_LOADS = TILE_CHUNKS / NT;
  static_assert(BN > 0 && BN % BstatWaveColumns == 0, "B-stat column tile");
  static_assert(TM % 16 == 0 && CH == 16 && TILE_CHUNKS % NT == 0 && A_LOADS >= 1, "tile shape");

  const int num_valid = size_m * top_k;
  const int ntp = num_tokens_post_pad[0];
  const int nblk = (ntp + TM - 1) / TM;
  const int blk0 = blockIdx.y * run_blocks;
  if (blk0 >= nblk)
    return;
  const int blk_end = min(blk0 + run_blocks, nblk);

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = __builtin_hcu_readfirstlane(tid >> 6);
  const int lrow = lane & 15;
  const int lk = lane >> 4;
  const int lq = lane >> 2;
  const int groups = size_k / GS;
  const int lgrp = scale_lane_group<GS>(lk);
  const int b_row_bytes = size_k / 2;
  const int b_perm_addr = (4 * lrow + lk) * 4;
  // output transpose: destination lane 4 j + c takes row j, chunk c from source lane j + 16 c
  const int t_perm_addr = ((lane & 3) * 16 + (lane >> 2)) * 4;
  const int n_warp = blockIdx.x * BN + warp * 16 * NN;

  // LDS: two A tile buffers, then two routing buffers of TM token rows and TM topk weights (separate
  // scalar arrays: aicc drops the element offset when it narrows a ds_read of an i32x2 to one lane)
  extern __shared__ i32x4 lds_a[];
  int *lds_tok = reinterpret_cast<int *>(lds_a + 2 * TILE_CHUNKS);
  float *lds_w = reinterpret_cast<float *>(lds_tok + 2 * TM);
  const __amdgpu_buffer_rsrc_t rsrc_a = make_rsrc(input);
  const __amdgpu_buffer_rsrc_t rsrc_o = make_rsrc(output);

  // A gather slots: slot i of thread tid = 16 byte chunk (i * NT + tid) of the row-major tile
  int a_row[A_LOADS], a_chunk[A_LOADS], a_lds[A_LOADS];
#pragma unroll
  for (int i = 0; i < A_LOADS; i++)
  {
    const int slot = i * NT + tid;
    a_row[i] = slot / (KS * CH);
    a_chunk[i] = slot % (KS * CH); // chunk of the 512 B row: step = chunk / CH
    const int step = a_chunk[i] / CH, c = a_chunk[i] % CH;
    a_lds[i] = step * STEP_CHUNKS + ((a_row[i] / 16) * CH + c) * 16 + ((a_row[i] & 15) ^ a_swz<OPT, CH>(c));
  }
  auto load_tok = [&](int (&tok)[A_LOADS], int blk) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
      tok[i] = sorted_token_ids[blk * TM + a_row[i]];
  };
  auto load_a = [&](i32x4 (&regs)[A_LOADS], const int (&tok)[A_LOADS]) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
    {
      const int src = tok[i] < num_valid ? tok[i] / top_k : 0;
      regs[i] = __builtin_hcu_raw_buffer_load_b128(rsrc_a, src * size_k * (int)sizeof(T) + a_chunk[i] * 16, 0, 0);
    }
  };
  auto store_a = [&](const i32x4 (&regs)[A_LOADS], int buf) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
    {
      i32x4 v = regs[i];
      if constexpr (!std::is_same_v<T, __hip_bfloat16>)
      {
        const uint32_t x0 = v[0], x1 = v[1], x2 = v[2], x3 = v[3];
        v[0] = (int)__builtin_amdgcn_perm(x2, x0, 0x05040100u);
        v[1] = (int)__builtin_amdgcn_perm(x2, x0, 0x07060302u);
        v[2] = (int)__builtin_amdgcn_perm(x3, x1, 0x05040100u);
        v[3] = (int)__builtin_amdgcn_perm(x3, x1, 0x07060302u);
      }
      lds_a[buf * TILE_CHUNKS + a_lds[i]] = v;
    }
  };
  // the thread holding the first chunk of a row also fetches and publishes the row's routing data
  auto load_w = [&](float (&w)[A_LOADS], const int (&tok)[A_LOADS]) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
      w[i] = (a_chunk[i] == 0 && tok[i] < num_valid) ? topk_weights[tok[i]] : 1.f;
  };
  auto store_tok = [&](const int (&tok)[A_LOADS], int buf) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
      if (a_chunk[i] == 0)
        lds_tok[buf * TM + a_row[i]] = tok[i];
  };
  auto store_w = [&](const float (&w)[A_LOADS], int buf) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
      if (a_chunk[i] == 0)
        lds_w[buf * TM + a_row[i]] = w[i];
  };

  // B: row j of n-tile t is output column n_warp + 4 NN (j % 4) + 4 t + j / 4 (see the kernel above)
  auto wcol = [&](int t, int j) { return n_warp + 4 * NN * (j & 3) + 4 * t + (j >> 2); };
  i32x4 braw[NN][KS];
  uint32_t sraw[NN];
  auto load_b = [&](int e) {
    const __amdgpu_buffer_rsrc_t rb = make_rsrc(qweight + (size_t)e * size_n * (size_k / 8));
    const __amdgpu_buffer_rsrc_t rs = make_rsrc(weight_scale + (size_t)e * size_n * groups);
#pragma unroll
    for (int t = 0; t < NN; t++)
    {
#pragma unroll
      for (int s = 0; s < KS; s++)
        braw[t][s] = __builtin_hcu_raw_buffer_load_b128(rb, wcol(t, lq) * b_row_bytes + (lane & 3) * 16 + s * (BK / 2), 0, 0);
      // scales of the lane's group of both K steps: lgrp * KS + s, s = 0, 1, in one dword
      sraw[t] = (uint32_t)__builtin_hcu_raw_buffer_load_b32(rs, (wcol(t, lrow) * groups + lgrp * KS) * (int)sizeof(T), 0, 0);
    }
  };
  i32x4 bq[NN][KS][4];
  auto dequant_b = [&]() {
#pragma unroll
    for (int t = 0; t < NN; t++)
#pragma unroll
      for (int s = 0; s < KS; s++)
      {
        i32x4 r = braw[t][s];
#pragma unroll
        for (int d = 0; d < 4; d++)
          r[d] = __builtin_amdgcn_ds_bpermute(b_perm_addr, r[d]);
        const DeqScale ds = make_scale<T>((sraw[t] >> (16 * s)) & 0xffffu);
#pragma unroll
        for (int u = 0; u < 4; u++)
          bq[t][s][u] = dequant8<T>((uint32_t)r[u], ds);
      }
  };

  f32x4 zq = {0.f, 0.f, 0.f, 0.f};
  asm volatile("" : "+v"(zq));
  f32x4 acc[NM][NN];
  auto compute = [&](int buf) {
    const i32x4 *a_tile = lds_a + buf * TILE_CHUNKS;
    auto load_frag = [&](i32x4 (&a)[NM], int s, int u) {
      const int c = 4 * lk + u;
#pragma unroll
      for (int mt = 0; mt < NM; mt++)
        a[mt] = a_tile[s * STEP_CHUNKS + (mt * CH + c) * 16 + (lrow ^ a_swz<OPT, CH>(c))];
    };
    i32x4 a_cur[NM], a_nxt[NM];
    load_frag(a_cur, 0, 0);
#pragma unroll
    for (int su = 0; su < KS * 4; su++)
    {
      const int s = su / 4, u = su % 4;
      if (su + 1 < KS * 4)
        load_frag(a_nxt, (su + 1) / 4, (su + 1) % 4);
#pragma unroll
      for (int t = 0; t < NN; t++)
      {
        const i32x4 b = bq[t][s][u];
        const i32x2 b0 = {b[0], b[1]}, b1 = {b[2], b[3]};
#pragma unroll
        for (int mt = 0; mt < NM; mt++)
        {
          const i32x2 a0 = {a_cur[mt][0], a_cur[mt][1]}, a1 = {a_cur[mt][2], a_cur[mt][3]};
          acc[mt][t] = mmac<T>(a0, b0, su == 0 ? zq : acc[mt][t]);
          acc[mt][t] = mmac<T>(a1, b1, acc[mt][t]);
        }
      }
#pragma unroll
      for (int mt = 0; mt < NM; mt++)
        a_cur[mt] = a_nxt[mt];
    }
  };
  auto epilogue = [&](int buf) {
    const int *rtok = lds_tok + buf * TM;
    const float *rw = lds_w + buf * TM;
#pragma unroll
    for (int mt = 0; mt < NM; mt++)
    {
      const float w = rw[mt * 16 + lrow];
      const f32x2 w2 = {w, w};
      uint32_t packed[4];
#pragma unroll
      for (int t = 0; t < NN; t++)
      {
        const f32x2 lo = f32x2{acc[mt][t][0], acc[mt][t][1]} * w2;
        const f32x2 hi = f32x2{acc[mt][t][2], acc[mt][t][3]} * w2;
        if constexpr ((OPT & OPT_RHU) != 0 && std::is_same_v<T, __hip_bfloat16>)
        {
          packed[2 * t] = pack2_bf16_rhu(lo[0], lo[1]);
          packed[2 * t + 1] = pack2_bf16_rhu(hi[0], hi[1]);
        }
        else
        {
          packed[2 * t] = pack2<T>(lo[0], lo[1]);
          packed[2 * t + 1] = pack2<T>(hi[0], hi[1]);
        }
      }
      i32x4 v;
#pragma unroll
      for (int d = 0; d < 4; d++)
        v[d] = __builtin_amdgcn_ds_bpermute(t_perm_addr, (int)packed[d]);
      const int tok = rtok[mt * 16 + (lane >> 2)];
      // padding rows get an offset past num_records and are dropped by the buffer store: no
      // divergent branch around the stores (it makes the waitcnt pass wait for their acks)
      const int off = tok < num_valid ? (tok * size_n + n_warp + 8 * (lane & 3)) * (int)sizeof(T) : 0x7fffffff;
      __builtin_hcu_raw_buffer_store_b128(v, rsrc_o, off, 0, 2 /* slc: streamed once by moe_sum */);
    }
  };

  // ---- pipeline ------------------------------------------------------------------------------
  // Entering the step of tile blk: set `cur` holds tile blk + 1 (A rows and topk weights in flight,
  // issued one step earlier, token rows known), set `nxt` the token rows of tile blk + 2.  A step
  // issues A / weights of blk + 2 and the token rows of blk + 3, publishes tile blk + 1 in the other
  // LDS buffers after its MMACs and stores tile blk: the dependent token -> A / weight gathers get
  // two tile steps instead of one (the gathers, not the MMACs, bounded this kernel).  The two
  // register sets alternate by unrolling the loop twice: copying an in-flight set would wait for it.
  // Prefetches are unconditional with the tile clamped to the last one, so the waitcnt pass never
  // has to assume them absent; only the rare expert-switch B loads are conditional.
  struct Set
  {
    i32x4 a[A_LOADS];
    float w[A_LOADS];
    int t[A_LOADS];
  };
  const int last = nblk - 1;
  int e = __builtin_hcu_readfirstlane(expert_ids[blk0]);
  Set s0, s1;
  {
    int t[A_LOADS];
    float w[A_LOADS];
    i32x4 areg[A_LOADS];
    load_tok(t, blk0);
    load_b(e);
    load_a(areg, t);
    load_w(w, t);
    load_tok(s0.t, min(blk0 + 1, last));
    load_tok(s1.t, min(blk0 + 2, last));
    load_a(s0.a, s0.t);
    load_w(s0.w, s0.t);
    store_a(areg, 0);
    store_tok(t, 0);
    store_w(w, 0);
    dequant_b();
    lds_barrier();
  }
  auto step = [&](int blk, Set &cur, Set &nxt) {
    const int buf = (blk - blk0) & 1;
    const int e_next = __builtin_hcu_readfirstlane(expert_ids[min(blk + 1, last)]);
    const bool switch_b = blk + 1 < blk_end && e_next != e;
    if (switch_b)
      load_b(e_next);
    load_a(nxt.a, nxt.t); // tile blk + 2
    load_w(nxt.w, nxt.t);
    store_tok(cur.t, buf ^ 1); // tile blk + 1
    load_tok(cur.t, min(blk + 3, last));
    compute(buf);
    store_a(cur.a, buf ^ 1);
    store_w(cur.w, buf ^ 1);
    epilogue(buf);
    if (switch_b)
    {
      dequant_b();
      e = e_next;
    }
    lds_barrier();
  };
  for (int blk = blk0; blk < blk_end; blk += 2)
  {
    step(blk, s0, s1);
    if (blk + 1 < blk_end)
      step(blk + 1, s1, s0);
  }
}

} // namespace w4a16_prefill

// Each CTA computes BM rows by BN physical columns per chunk. NLoop > 1 reuses
// A across chunks when the full reduction fits in LDS (size_k == 256).
template <int BM, int BN, int NLoop, W4A16Pipeline Pipeline, W4A16Epilogue Epilogue, typename T, int GS>
void launch_moe_w4a16_prefill(const GemmParams_w4a16<T> &p)
{
  using Config = w4a16_prefill::Tile<BM, BN, Pipeline>;
  constexpr int WARPS = Config::WARPS, ST = Config::ST, AH = Config::AH;
  constexpr bool GEMM2 = Epilogue == W4A16Epilogue::DownWeighted, SILU = Epilogue == W4A16Epilogue::UpSilu;
  TORCH_CHECK(p.size_k > 0 && p.size_k % (2 * w4a16_prefill::BK) == 0, "moe_w4a16 prefill: size_k=", p.size_k,
              " must be a positive multiple of ", 2 * w4a16_prefill::BK);
  if constexpr (NLoop > 1)
  {
    // The resident pipeline consumes exactly two K steps and NLoop full column chunks.
    if (p.size_k != 2 * w4a16_prefill::BK || p.size_n % (BN * NLoop) != 0)
    {
      launch_moe_w4a16_prefill<BM, BN, 1, Pipeline, Epilogue, T, GS>(p);
      return;
    }
  }
  static_assert(NLoop > 0 && Config::NN % 2 == 0, "valid NLoop and 16-byte stores");
  TORCH_CHECK(p.size_n % (BN * NLoop) == 0, "moe_w4a16 prefill: size_n=", p.size_n, " must be a multiple of ", BN * NLoop);

  static_assert(ST == 1 || ST == 2, "LDS stages");
  static_assert(NLoop == 1 || ST == 2, "resident-A mode keeps both K halves in LDS");
  static_assert(!SILU || NLoop == 1, "fused silu_and_mul is a gemm1 streamed-A epilogue");
  const int lds_bytes = ST * BM * w4a16_prefill::BK * (int)sizeof(T) / AH;
  dim3 grid(p.size_n / (BN * NLoop), DIVIDE(p.sorted_token_lens, BM), 1);
  dim3 block(WARPS * 64, 1, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  if constexpr (GEMM2)
    TORCH_CHECK(p.topk_weights != nullptr, "moe_w4a16 prefill gemm2 needs topk_weights");
  w4a16_prefill::moe_w4a16_prefill_kernel<T, BM, BN, NLoop, Pipeline, Epilogue, GS><<<grid, block, lds_bytes, stream>>>(
      p.ptr_A, p.ptr_B0, p.ptr_C, p.ptr_B_scale, GEMM2 ? p.topk_weights : nullptr, p.sorted_token_ids, p.expert_ids,
      p.num_tokens_post_pad_ptr, p.size_m, p.size_n, p.size_k, p.sorted_token_lens, p.top_k);
}

// gemm2 (size_k == 256) with the dequantized B in registers: TM rows per routing block (BLOCK_SIZE_M
// of the gemm2 config), WARPS * NN * 16 output columns per CTA, each CTA walks a run of routing
// blocks (targeting two waves of resident CTAs over the device).
template <int TM, int BN, typename T, int GS>
void launch_moe_w4a16_g2_bstat(const GemmParams_w4a16<T> &p)
{
  constexpr int WARPS = BN / w4a16_prefill::BstatWaveColumns;
  TORCH_CHECK(p.size_k == 2 * w4a16_prefill::BK, "moe_w4a16 gemm2 bstat: size_k must be ", 2 * w4a16_prefill::BK, ", got ", p.size_k);
  TORCH_CHECK(p.size_n > 0 && p.size_n % BN == 0, "moe_w4a16 gemm2 bstat: size_n=", p.size_n, " must be a positive multiple of ", BN);
  TORCH_CHECK(p.topk_weights != nullptr, "moe_w4a16 gemm2 bstat needs topk_weights");
  TORCH_CHECK(w4a16_bstat_output_fits(p),
              "moe_w4a16 gemm2 bstat: output must be below 2 GiB (32 bit buffer offsets)");
  const int nblk_ub = DIVIDE(p.sorted_token_lens, TM);
  if (nblk_ub == 0)
    return;
  // The W4A16 entry point guards the input device before dispatch.
  int dev = 0, num_cu = 0;
  TORCH_CHECK(hipGetDevice(&dev) == hipSuccess &&
                  hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, dev) == hipSuccess && num_cu > 0,
              "moe_w4a16 gemm2 bstat: cannot query the current device CU count");
  const int xs = p.size_n / BN;
  const int lds_bytes = 2 * 2 * TM * w4a16_prefill::BK * (int)sizeof(T) + 2 * TM * 8;
  // CTAs resident per CU (LDS and thread limits), two CTA waves over the device, >= 4 blocks per run
  const int ctas_per_cu = std::max(1, std::min(65536 / lds_bytes, 2560 / (WARPS * 64)));
  const int target_y = std::max(1, DIVIDE(2 * num_cu * ctas_per_cu, xs));
  const int run_blocks = std::max(4, DIVIDE(nblk_ub, target_y));
  dim3 grid(xs, DIVIDE(nblk_ub, run_blocks), 1);
  dim3 block(WARPS * 64, 1, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  w4a16_prefill::moe_w4a16_prefill_g2_bstat_kernel<T, TM, BN, GS><<<grid, block, lds_bytes, stream>>>(
      p.ptr_A, p.ptr_B0, p.ptr_C, p.ptr_B_scale, p.topk_weights, p.sorted_token_ids, p.expert_ids,
      p.num_tokens_post_pad_ptr, p.size_m, p.size_n, p.size_k, p.top_k, run_blocks);
}
