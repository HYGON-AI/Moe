// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// W4A16 MoE decode GEMM (gemm1 = gate/up, gemm2 = down) for small M: few tokens per routing
// block, so the kernel streams B and is bound by the B loads and the dequant VALU, not by MMAC.
//
// Layout: the tensors of the prefill kernels (moe_w4a16_prefill.h). Each block
// computes BM routing rows and BN physical weight columns (gate and up included).
//
// Block: WN x KW warps.  Warp (wn, wk) owns NN 16-row B tiles (NN * 16 output columns, or NN / 2 *
// 16 gate and as many up columns with SILU) and the K steps [wk SPW, (wk + 1) SPW) of 128.  A goes
// through a double-buffered LDS stage holding the KW K-step slabs of one pipeline step (one LDS
// barrier per step, conflict-free layout of the prefill kernel); B is loaded coalesced PD steps
// ahead and transposed to the MMAC layout with ds_bpermute.  The fp32 partial sums of the KW warps
// meet in LDS, which also transposes the tile for 16 byte output stores (silu_and_mul fused for
// gemm1, topk weight for gemm2).
//
// The MMAC operands are swapped (B rows = MMAC rows): lane l holds the results of B row l % 16 and
// the tokens 4 r + l / 16, so its four accumulators share the scale of the B row it dequantizes.
// Group 128 (one scale per row and K step) uses that to scale after the MMACs: B is converted
// without scale to BIAS + q (bf16 0x4300 | q = 128 + q, fp16 0x6400 | q = 1024 + q, one v_and_or
// per two values, instead of the fp32 convert / fma / pack of the scaled dequant), the per-step
// MMAC chain of every tile starts from -(BIAS + 8) * sum_k A (one MMAC chain against a constant
// operand per m-tile), so it yields sum_k A (q - 8) exactly, and two packed FMAs per tile apply the
// scale.  The weights are thus exact (q - 8) * s in fp32, not rounded to 16 bit as in the scaled
// dequant.  Group 32 uses the scaled dequant of the prefill kernel.

#pragma once

#include <hip/hip_runtime.h>
#include <type_traits>

#include "moe_w4a16_prefill.h"

namespace w4a16_decode
{

using w4a16_prefill::BK;
using w4a16_prefill::f32x2;
using w4a16_prefill::f32x4;
using w4a16_prefill::i32x2;
using w4a16_prefill::i32x4;
using w4a16_prefill::u32x2;

// 8 int4 (k0..k7, low nibble first) -> BIAS + q as 16 bit pairs (k0, k4), (k1, k5), (k2, k6), (k3, k7)
template <typename T>
__device__ __forceinline__ i32x4 dequant8_raw(uint32_t q)
{
  constexpr uint32_t E = std::is_same_v<T, __hip_bfloat16> ? 0x43004300u : 0x64006400u;
  i32x4 r;
  r[0] = (int)((q & 0x000f000fu) | E);
  r[1] = (int)(((q >> 4) & 0x000f000fu) | E);
  r[2] = (int)(((q >> 8) & 0x000f000fu) | E);
  r[3] = (int)(((q >> 12) & 0x000f000fu) | E);
  return r;
}

// packed -(BIAS + 8): bf16 -136 = 0xc308, fp16 -1032 = 0xe408 (both exact)
template <typename T>
constexpr uint32_t neg_bias2()
{
  return std::is_same_v<T, __hip_bfloat16> ? 0xc308c308u : 0xe408e408u;
}

template <typename T>
__device__ __forceinline__ float scale_f32(uint32_t bits)
{
  if constexpr (std::is_same_v<T, __hip_bfloat16>)
    return __builtin_bit_cast(float, bits << 16);
  else
    return (float)__builtin_bit_cast(_Float16, (uint16_t)bits);
}

template <int BM, int BN>
struct Tile
{
  static_assert(BM > 0 && BM % 16 == 0, "invalid W4A16 decode row tile");
  static_assert(BN == 64 || BN == 128 || BN == 256, "unsupported W4A16 decode column tile");
  static constexpr int NM = BM / 16;
  static constexpr int NN = 2;
  static constexpr int WN = BN / (NN * 16);
  static constexpr int KW = BN == 64 ? 2 : 1;
  static constexpr int Threads = WN * KW * 64;
};

template <typename T, int BM, int BN, int ReduceK, int PD, W4A16Epilogue Epilogue, int GS>
__global__ void __launch_bounds__((Tile<BM, BN>::Threads)) moe_w4a16_decode_kernel(
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
  using Config = Tile<BM, BN>;
  constexpr int NM = Config::NM, NN = Config::NN, WN = Config::WN, KW = Config::KW;
  static_assert(ReduceK % (KW * BK) == 0, "K must split evenly across waves");
  constexpr int SPW = ReduceK / (KW * BK);
  constexpr bool GEMM2 = Epilogue == W4A16Epilogue::DownWeighted, SILU = Epilogue == W4A16Epilogue::UpSilu;
  constexpr bool POST = GS == 128;
  constexpr int NT = Config::Threads;
  constexpr int NH = SILU ? NN / 2 : NN;      // output tiles of a warp
  constexpr int BNB = BN;                     // B rows of the block
  constexpr int BNO = WN * NH * 16;           // output columns of the block
  constexpr int CH = BK * (int)sizeof(T) / 16; // 16 byte A chunks per row and K step
  constexpr int SLAB = BM * CH;               // one K step of the block's rows
  constexpr int STAGE = KW * SLAB;            // one LDS stage: the KW slabs of a pipeline step
  constexpr int A_LOADS = (STAGE + NT - 1) / NT;
  constexpr int PST = BNB + 16;               // row stride of the fp32 partial tile in LDS
  constexpr int SW = SPW >= 2 ? SPW / 2 : 1;  // dwords of the lane's scales of SPW steps
  // A in (0,4),(1,5),(2,6),(3,7) order: fp16 dequant, and the raw 16 bit pairs of POST
  constexpr bool PERM_A = POST || !std::is_same_v<T, __hip_bfloat16>;
  static_assert(GS == 32 || GS == 128, "W4A16 group size along K");
  static_assert(CH == 16 && NN >= 1 && (!SILU || NN % 2 == 0) && PD >= 1 && PD <= SPW, "tile shape");
  static_assert(SPW == 1 || SPW == 2 || SPW % 4 == 0, "packed scale load width");
  static_assert(BNO % 8 == 0, "16 byte output chunks");

  const int m_block = blockIdx.y;
  const int num_valid = size_m * top_k;
  if (m_block * BM >= num_tokens_post_pad[0] || sorted_token_ids[m_block * BM] >= num_valid)
    return;
  const int expert = expert_ids[m_block];

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = __builtin_hcu_readfirstlane(tid >> 6);
  const int wn = warp % WN;
  const int wk = warp / WN;
  const int lrow = lane & 15;
  const int lk = lane >> 4;
  const int lq = lane >> 2;
  const int m_base = m_block * BM;
  // sorted id (topk slot) of row `row` of the block, >= num_valid for a padding row
  auto row_tok = [&](int row) -> int {
    return sorted_token_ids[min(m_base + row, sorted_token_lens - 1)];
  };
  const int steps = size_k / BK; // KW * SPW
  const int step0 = wk * SPW;     // first K step of the warp

  extern __shared__ i32x4 lds_a[];

  // ---- A gather: stage chunk i * NT + tid = slab w (K step w * SPW + s), row, chunk c ---------
  const __amdgpu_buffer_rsrc_t rsrc_a = w4a16_prefill::make_rsrc(input);
  int a_off[A_LOADS];
  int a_lds[A_LOADS];
#pragma unroll
  for (int i = 0; i < A_LOADS; i++)
  {
    const int slot = min(i * NT + tid, STAGE - 1);
    const int w = slot / SLAB;
    const int row = (slot % SLAB) / CH;
    const int c = slot % CH;
    const int tok = row_tok(row);
    const int src = tok < num_valid ? tok / top_k : 0;
    a_off[i] = src * size_k * (int)sizeof(T) + w * SPW * BK * (int)sizeof(T) + c * 16;
    a_lds[i] = w * SLAB + ((row / 16) * CH + c) * 16 + ((row & 15) ^ w4a16_prefill::a_swz<w4a16_prefill::OPT_SWZ, CH>(c));
  }
  auto load_a = [&](i32x4 (&regs)[A_LOADS], int s) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
      if (A_LOADS * NT == STAGE || i * NT + tid < STAGE)
        regs[i] = __builtin_hcu_raw_buffer_load_b128(rsrc_a, a_off[i] + s * BK * (int)sizeof(T), 0, 0);
  };
  auto store_a = [&](const i32x4 (&regs)[A_LOADS], int stage) {
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
    {
      if (A_LOADS * NT == STAGE || i * NT + tid < STAGE)
      {
        i32x4 v = regs[i];
        if constexpr (PERM_A)
        {
          const uint32_t x0 = v[0], x1 = v[1], x2 = v[2], x3 = v[3];
          v[0] = (int)__builtin_amdgcn_perm(x2, x0, 0x05040100u);
          v[1] = (int)__builtin_amdgcn_perm(x2, x0, 0x07060302u);
          v[2] = (int)__builtin_amdgcn_perm(x3, x1, 0x05040100u);
          v[3] = (int)__builtin_amdgcn_perm(x3, x1, 0x07060302u);
        }
        lds_a[stage * STAGE + a_lds[i]] = v;
      }
    }
  };

  // ---- B / scales: row j of tile t is output column n_warp + 16 (t % NH) + j (+ the up rows) ---
  const int groups = size_k / GS;
  const __amdgpu_buffer_rsrc_t rsrc_b = w4a16_prefill::make_rsrc(qweight + (size_t)expert * size_n * (size_k / 8));
  const __amdgpu_buffer_rsrc_t rsrc_s = w4a16_prefill::make_rsrc(weight_scale + (size_t)expert * size_n * groups);
  const int b_row_bytes = size_k / 2;
  const int b_perm_addr = (4 * lrow + lk) * 4;
  const int up_rows = SILU ? size_n / 2 : 0;
  const int n_warp = blockIdx.x * BNO + wn * NH * 16;
  auto wcol = [&](int t, int j) { return n_warp + 16 * (t % NH) + j + (t >= NH ? up_rows : 0); };
  auto load_b = [&](i32x4 (&regs)[NN], int step) {
#pragma unroll
    for (int t = 0; t < NN; t++)
      regs[t] = __builtin_hcu_raw_buffer_load_b128(rsrc_b, wcol(t, lq) * b_row_bytes + (lane & 3) * 16 + step * (BK / 2), 0, 0);
  };
  auto permute_b = [&](i32x4 (&regs)[NN]) {
#pragma unroll
    for (int t = 0; t < NN; t++)
#pragma unroll
      for (int d = 0; d < 4; d++)
        regs[t][d] = __builtin_amdgcn_ds_bpermute(b_perm_addr, regs[t][d]);
  };
  // scales of the lane's B row (lrow) for the warp's SPW steps: consecutive in the shuffled layout
  const int lgrp = w4a16_prefill::scale_lane_group<GS>(lk);
  uint32_t sreg[NN][SW];
  auto load_scales = [&]() {
#pragma unroll
    for (int t = 0; t < NN; t++)
    {
      const int off = (wcol(t, lrow) * groups + lgrp * steps + step0) * (int)sizeof(T);
      if constexpr (SPW == 1)
        sreg[t][0] = (uint32_t)__builtin_hcu_raw_buffer_load_b16(rsrc_s, off, 0, 0);
      else if constexpr (SPW == 2)
        sreg[t][0] = (uint32_t)__builtin_hcu_raw_buffer_load_b32(rsrc_s, off, 0, 0);
      else if constexpr (SPW == 4)
      {
        const u32x2 v = __builtin_bit_cast(u32x2, __builtin_hcu_raw_buffer_load_b64(rsrc_s, off, 0, 0));
        sreg[t][0] = v[0];
        sreg[t][1] = v[1];
      }
      else
      {
#pragma unroll
        for (int q = 0; q < SW / 4; q++)
        {
          const i32x4 v = __builtin_hcu_raw_buffer_load_b128(rsrc_s, off + 16 * q, 0, 0);
#pragma unroll
          for (int d = 0; d < 4; d++)
            sreg[t][4 * q + d] = (uint32_t)v[d];
        }
        if constexpr (SPW % 8 == 4)
        {
          constexpr int tail = 4 * (SW / 4);
          const u32x2 v = __builtin_bit_cast(u32x2, __builtin_hcu_raw_buffer_load_b64(rsrc_s, off + 4 * tail, 0, 0));
          sreg[t][tail] = v[0];
          sreg[t][tail + 1] = v[1];
        }
      }
    }
  };
  auto scale_bits = [&](int t, int s) { return (sreg[t][s >> 1] >> ((s & 1) * 16)) & 0xffffu; };

  // ---- one K step: B tiles (MMAC layout), A stage, step s of the warp ---------------------------
  f32x4 acc[NM][NN];
#pragma unroll
  for (int mt = 0; mt < NM; mt++)
#pragma unroll
    for (int t = 0; t < NN; t++)
      acc[mt][t] = f32x4{0.f, 0.f, 0.f, 0.f};
  f32x4 zq = {0.f, 0.f, 0.f, 0.f};
  asm volatile("" : "+v"(zq));
  auto compute = [&](int stage, const i32x4 (&breg)[NN], int s) {
    const i32x4 *a_stage = lds_a + stage * STAGE + wk * SLAB;
    i32x4 a[NM][4];
#pragma unroll
    for (int mt = 0; mt < NM; mt++)
#pragma unroll
      for (int u = 0; u < 4; u++)
      {
        const int c = 4 * lk + u;
        a[mt][u] = a_stage[(mt * CH + c) * 16 + (lrow ^ w4a16_prefill::a_swz<w4a16_prefill::OPT_SWZ, CH>(c))];
      }
    if constexpr (POST)
    {
      // -(BIAS + 8) * sum_k A of the step for the lane's tokens: the start of every tile's chain
      const i32x2 nb = {(int)neg_bias2<T>(), (int)neg_bias2<T>()};
      f32x4 rs[NM];
#pragma unroll
      for (int mt = 0; mt < NM; mt++)
      {
        rs[mt] = zq;
#pragma unroll
        for (int u = 0; u < 4; u++)
        {
          rs[mt] = w4a16_prefill::mmac<T>(nb, i32x2{a[mt][u][0], a[mt][u][1]}, rs[mt]);
          rs[mt] = w4a16_prefill::mmac<T>(nb, i32x2{a[mt][u][2], a[mt][u][3]}, rs[mt]);
        }
      }
#pragma unroll
      for (int t = 0; t < NN; t++)
      {
        const float sc = scale_f32<T>(scale_bits(t, s));
        const f32x2 s2 = {sc, sc};
        f32x4 p[NM];
#pragma unroll
        for (int mt = 0; mt < NM; mt++)
          p[mt] = rs[mt];
#pragma unroll
        for (int u = 0; u < 4; u++)
        {
          const i32x4 b = dequant8_raw<T>((uint32_t)breg[t][u]);
#pragma unroll
          for (int mt = 0; mt < NM; mt++)
          {
            p[mt] = w4a16_prefill::mmac<T>(i32x2{b[0], b[1]}, i32x2{a[mt][u][0], a[mt][u][1]}, p[mt]);
            p[mt] = w4a16_prefill::mmac<T>(i32x2{b[2], b[3]}, i32x2{a[mt][u][2], a[mt][u][3]}, p[mt]);
          }
        }
#pragma unroll
        for (int mt = 0; mt < NM; mt++)
        {
          const f32x2 lo = __builtin_elementwise_fma(f32x2{p[mt][0], p[mt][1]}, s2, f32x2{acc[mt][t][0], acc[mt][t][1]});
          const f32x2 hi = __builtin_elementwise_fma(f32x2{p[mt][2], p[mt][3]}, s2, f32x2{acc[mt][t][2], acc[mt][t][3]});
          acc[mt][t] = f32x4{lo[0], lo[1], hi[0], hi[1]};
        }
      }
    }
    else
    {
#pragma unroll
      for (int t = 0; t < NN; t++)
      {
        const w4a16_prefill::DeqScale ds = w4a16_prefill::make_scale<T>(scale_bits(t, s));
#pragma unroll
        for (int u = 0; u < 4; u++)
        {
          const i32x4 b = w4a16_prefill::dequant8<T>((uint32_t)breg[t][u], ds);
#pragma unroll
          for (int mt = 0; mt < NM; mt++)
          {
            acc[mt][t] = w4a16_prefill::mmac<T>(i32x2{b[0], b[1]}, i32x2{a[mt][u][0], a[mt][u][1]}, acc[mt][t]);
            acc[mt][t] = w4a16_prefill::mmac<T>(i32x2{b[2], b[3]}, i32x2{a[mt][u][2], a[mt][u][3]}, acc[mt][t]);
          }
        }
      }
    }
  };

  // ---- pipeline: B PD steps ahead, A one step ahead through the two LDS stages ------------------
  i32x4 areg[A_LOADS];
  i32x4 bring[PD][NN];
  load_a(areg, 0);
#pragma unroll
  for (int q = 0; q < PD; q++)
    load_b(bring[q], step0 + q);
  load_scales();
  store_a(areg, 0);
  if constexpr (SPW > 1)
    load_a(areg, 1);
  w4a16_prefill::lds_barrier();
#pragma unroll
  for (int s = 0; s < SPW; s++)
  {
    i32x4 (&bc)[NN] = bring[s % PD];
    permute_b(bc);
    compute(s & 1, bc, s);
    if (s + PD < SPW)
      load_b(bc, step0 + s + PD);
    if (s + 1 < SPW)
    {
      store_a(areg, (s + 1) & 1);
      if (s + 2 < SPW)
        load_a(areg, s + 2);
      w4a16_prefill::lds_barrier();
    }
  }

  // ---- partial tiles of the KW warps -> LDS [wk][row][B row slot], then 16 byte output chunks ----
  w4a16_prefill::lds_barrier(); // every warp is done with the A stages the partials overwrite
  float *part = reinterpret_cast<float *>(lds_a);
#pragma unroll
  for (int mt = 0; mt < NM; mt++)
#pragma unroll
    for (int t = 0; t < NN; t++)
#pragma unroll
      for (int r = 0; r < 4; r++)
        part[(wk * BM + mt * 16 + 4 * r + lk) * PST + wn * NN * 16 + t * 16 + lrow] = acc[mt][t][r];
  w4a16_prefill::lds_barrier();

  const int n_out = SILU ? size_n / 2 : size_n;
  for (int e = tid; e < BM * BNO / 8; e += NT)
  {
    const int row = e / (BNO / 8);
    const int cc = (e % (BNO / 8)) * 8; // output column inside the block
    const int tok = row_tok(row);
    if (tok >= num_valid)
      continue;
    // output column cc + i of warp w = (cc + i) / (NH * 16): gate slot w * NN * 16 + (cc + i) % (NH * 16)
    const int w = cc / (NH * 16);
    const int slot = w * NN * 16 + cc % (NH * 16);
    f32x4 g0 = {0.f, 0.f, 0.f, 0.f}, g1 = g0, u0 = g0, u1 = g0;
#pragma unroll
    for (int k = 0; k < KW; k++)
    {
      const f32x4 *src = reinterpret_cast<const f32x4 *>(part + (k * BM + row) * PST + slot);
      g0 += src[0];
      g1 += src[1];
      if constexpr (SILU)
      {
        u0 += src[NH * 4];
        u1 += src[NH * 4 + 1];
      }
    }
    float v[8] = {g0[0], g0[1], g0[2], g0[3], g1[0], g1[1], g1[2], g1[3]};
    if constexpr (SILU)
    {
      const float up[8] = {u0[0], u0[1], u0[2], u0[3], u1[0], u1[1], u1[2], u1[3]};
#pragma unroll
      for (int i = 0; i < 8; i += 2)
      {
        const f32x2 r = w4a16_prefill::silu_mul2<T>(f32x2{v[i], v[i + 1]}, f32x2{up[i], up[i + 1]});
        v[i] = r[0];
        v[i + 1] = r[1];
      }
    }
    if constexpr (GEMM2)
    {
      const float wt = topk_weights[tok];
#pragma unroll
      for (int i = 0; i < 8; i++)
        v[i] *= wt;
    }
    const i32x4 o = {(int)w4a16_prefill::pack2<T>(v[0], v[1]), (int)w4a16_prefill::pack2<T>(v[2], v[3]),
                     (int)w4a16_prefill::pack2<T>(v[4], v[5]), (int)w4a16_prefill::pack2<T>(v[6], v[7])};
    *reinterpret_cast<i32x4 *>(output + (size_t)tok * n_out + blockIdx.x * BNO + cc) = o;
  }
}

// Select fully unrolled kernels with the existing packed scale loads.
template <bool GEMM2, typename Launch>
void dispatch_decode_k(int size_k, Launch launch)
{
  if constexpr (GEMM2)
  {
    switch (size_k)
    {
    case 256: return launch(std::integral_constant<int, 256>{});
    case 512: return launch(std::integral_constant<int, 512>{});
    case 1024: return launch(std::integral_constant<int, 1024>{});
    case 2048: return launch(std::integral_constant<int, 2048>{});
    default: TORCH_CHECK(false, "moe_w4a16 decode GEMM2 expects K in {256, 512, 1024, 2048}, got ", size_k);
    }
  }
  else
  {
    switch (size_k)
    {
    case 2048: return launch(std::integral_constant<int, 2048>{});
    case 1024: return launch(std::integral_constant<int, 1024>{});
    case 3072: return launch(std::integral_constant<int, 3072>{});
    case 4096: return launch(std::integral_constant<int, 4096>{});
    case 6144: return launch(std::integral_constant<int, 6144>{});
    case 7168: return launch(std::integral_constant<int, 7168>{});
    default: TORCH_CHECK(false, "moe_w4a16 decode GEMM1 expects K in {1024, 2048, 3072, 4096, 6144, 7168}, got ", size_k);
    }
  }
}

} // namespace w4a16_decode

// BK is fixed at 128 per wave. The supported reduction length determines the
// steps per wave and caps the requested B prefetch depth.
template <int BM, int BN, int PrefetchDepth, W4A16Epilogue Epilogue, typename T, int GS>
void launch_moe_w4a16_decode(const GemmParams_w4a16<T> &p)
{
  using Config = w4a16_decode::Tile<BM, BN>;
  constexpr int KW = Config::KW, BK = w4a16_decode::BK;
  constexpr bool GEMM2 = Epilogue == W4A16Epilogue::DownWeighted, SILU = Epilogue == W4A16Epilogue::UpSilu;
  constexpr int BNO = SILU ? BN / 2 : BN;
  static_assert(PrefetchDepth > 0, "W4A16 decode requires a positive prefetch depth");
  const int n_out = SILU ? p.size_n / 2 : p.size_n;
  TORCH_CHECK(p.size_n > 0 && p.size_n % BN == 0,
              "moe_w4a16 decode: size_n=", p.size_n, " must be a positive multiple of ", BN);
  TORCH_CHECK(!GEMM2 || p.topk_weights != nullptr, "moe_w4a16 decode gemm2 needs topk_weights");
  constexpr int lds_bytes = std::max(2 * KW * BM * BK * (int)sizeof(T), KW * BM * (BN + 16) * 4);
  static_assert(lds_bytes <= 65536, "moe_w4a16 decode: LDS (A stages / split-K partials) above 64 KB");
  // the valid routing blocks come first and hold at least one row each: at most size_m * top_k of
  // them, far fewer than sorted_token_lens / BM for small M (empty blocks only cost their launch)
  const int64_t capacity = (static_cast<int64_t>(p.sorted_token_lens) + BM - 1) / BM;
  const int m_blocks = std::min(static_cast<int64_t>(p.size_m) * p.top_k, capacity);
  if (m_blocks == 0)
    return;
  dim3 grid(n_out / BNO, m_blocks, 1);
  dim3 block(Config::Threads, 1, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  w4a16_decode::dispatch_decode_k<GEMM2>(p.size_k, [&](auto k_tag) {
    constexpr int ReduceK = decltype(k_tag)::value;
    constexpr int depth = std::min(PrefetchDepth, ReduceK / (KW * BK));
    w4a16_decode::moe_w4a16_decode_kernel<T, BM, BN, ReduceK, depth, Epilogue, GS><<<grid, block, lds_bytes, stream>>>(
        p.ptr_A, p.ptr_B0, p.ptr_C, p.ptr_B_scale, GEMM2 ? p.topk_weights : nullptr, p.sorted_token_ids, p.expert_ids,
        p.num_tokens_post_pad_ptr, p.size_m, p.size_n, p.size_k, p.sorted_token_lens, p.top_k);
  });
}
