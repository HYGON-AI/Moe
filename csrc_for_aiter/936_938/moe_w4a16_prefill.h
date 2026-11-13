// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
//
// W4A16 MoE prefill GEMM (gemm1 = gate/up, gemm2 = down) for large M.
//
// Layout (same tensors as the decode kernels in moe_w4a16_opt.h):
//   input   [rows, size_k]                 bf16/fp16, rows gathered through sorted_token_ids
//   qweight [E, size_n, size_k / 8]        uint32, 8 int4 along K per word (low nibble first)
//   scale   [E, size_n, size_k / 32]       group 32 along K, w4a16_marlin_scale shuffled
//   output  [size_m * top_k, size_n]       row = sorted token id
// Symmetric quantization only (zero point fixed at 8, qzeros ignored), like the decode kernels.
//
// Tiling: every warp owns all BM rows of the block and NN * 16 output columns, so a dequantized
// B fragment is reused by all BM / 16 m-tiles (dequant VALU does not overlap with MMAC on gfx936,
// so this reuse is what bounds the kernel).  A goes through an LDS tile of BM x 128 per K step
// (double-buffered: one barrier per step, or single-buffered: two barriers, half the LDS); B is
// loaded coalesced one step ahead and transposed to the MMAC layout with ds_bpermute.  With
// size_k == 256 (gemm2) the whole A tile stays in LDS and a block walks NCH column chunks.
//
// MMAC 16x16x16 operands: lane l supplies row l % 16 and k 4 * (l / 16) + [0, 4); the result
// register r of lane l is C[l % 16][4 * r + l / 16].  Inside one 128-wide K step lane group
// c = l / 16 covers the contiguous k range [32 c, 32 c + 32) (one scale group) for both A and B.
// B row j of n-tile t is the output column 4 * NN * (j % 4) + 4 * t + j / 4 of the warp, which
// makes the 4 * NN accumulator values of a lane contiguous output columns.

#pragma once

#include <hip/hip_runtime.h>
#include <type_traits>

namespace w4a16_prefill
{

using f32x2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using i32x2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using i32x4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using u32x2 = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;

constexpr int BK = 128;  // K per pipeline step
constexpr int GROUP = 32; // quantization group along K

// Barrier that only waits for LDS traffic.  __syncthreads() also drains vmcnt, which would
// serialize the global prefetch issued right before it.
__device__ __forceinline__ void lds_barrier()
{
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void *ptr)
{
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(ptr), (short)0, 0x7fffffff, 0x00020000);
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

template <typename T, int BM, int WARPS, int NN, int NCH, int ST, bool MUL_TOPK, bool PREFETCH_A, int WPE, int SCM, bool SILU,
          int AH>
__global__ void __launch_bounds__(WARPS * 64) __attribute__((amdgpu_waves_per_eu(WPE > 0 ? WPE : (ST == 1 ? 2 * WARPS / 4 : 1), 8))) moe_w4a16_prefill_kernel(
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
  constexpr int NT = WARPS * 64;
  constexpr int NM = BM / 16;
  constexpr int WARP_N = NN * 16;
  constexpr int BN = WARPS * WARP_N;
  constexpr int ROW_CHUNKS = BK * (int)sizeof(T) / 16; // 16-byte chunks per A row and K step
  // AH = 2: A goes through LDS in two half steps per K step (half the LDS per stage).  Half h
  // holds, for every lane group lk, the k range [32 lk + 16 h, 32 lk + 16 h + 16), i.e. sub-steps
  // u = 2 h, 2 h + 1 of the compute loop, so B, the scales and the dequant stay per 128-K step.
  constexpr int CHUNKS = ROW_CHUNKS / AH; // A chunks per row in one LDS stage
  constexpr int STAGE_CHUNKS = BM * CHUNKS;
  constexpr int A_LOADS = (STAGE_CHUNKS + NT - 1) / NT;
  static_assert(BM % 16 == 0 && ROW_CHUNKS == 16 && NT % 16 == 0, "tile shape");
  static_assert(AH == 1 || (AH == 2 && ST == 2 && NCH == 1), "half-step A staging is a streamed ST2 pipeline");
  const int m_block = blockIdx.y;
  const int num_valid = size_m * top_k;
  if (m_block * BM >= num_tokens_post_pad[0])
    return;
  if (sorted_token_ids[m_block * BM] >= num_valid)
    return;

  const int tid = threadIdx.x;
  const int lane = tid & 63;
  const int warp = __builtin_amdgcn_readfirstlane(tid >> 6);
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
    a_lds[i] = ((row / 16) * CHUNKS + chunk) * 16 + ((row & 15) ^ (chunk & 7));
  }

  // ---- B / scale setup ----------------------------------------------------------------------
  // B row j of n-tile t is output column n_warp + 4 NN (j % 4) + 4 t + j / 4.  B is loaded with 4
  // adjacent lanes per row (64 contiguous bytes, lane 4 j + c holds chunk c of row j) and then
  // transposed with ds_bpermute to the MMAC layout (lane j + 16 c).
  const int groups = size_k / GROUP;
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
  int n_warp = (blockIdx.x * NCH) * (BN / (NN / NH)) + warp * 4 * 4 * NH;
  // B row of row j of n-tile t
  auto wcol = [&](int t, int j) {
    return n_warp + 4 * NH * (j & 3) + 4 * (t % NH) + (j >> 2) + (t >= NH ? up_rows : 0);
  };
  auto b_offset = [&](int t, int step) {
    return wcol(t, lq) * b_row_bytes + (lane & 3) * 16 + step * (BK / 2);
  };
  // shuffled scale position of group g = 4 * step + lk is lk * steps + step
  auto s_offset = [&](int t, int step) {
    return (wcol(t, lrow) * groups + lk * steps + step) * (int)sizeof(T);
  };

  // q = K step (AH == 1) or half step (AH == 2)
  auto load_a = [&](i32x4(&regs)[A_LOADS], int q) {
    const int off = (q / AH) * BK * (int)sizeof(T) + (q % AH) * 32;
#pragma unroll
    for (int i = 0; i < A_LOADS; i++)
      if (A_LOADS * NT == STAGE_CHUNKS || i * NT + tid < STAGE_CHUNKS)
        regs[i] = __builtin_amdgcn_raw_buffer_load_b128(rsrc_a, a_off[i] + off, 0, 0);
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
      regs[t] = __builtin_amdgcn_raw_buffer_load_b128(rsrc_b, b_offset(t, step), 0, 0);
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
        regs[t] = __builtin_bit_cast(u32x2, __builtin_amdgcn_raw_buffer_load_b64(rsrc_s, s_offset(t, step), 0, 0));
      else
        regs[t][0] = (uint32_t)__builtin_amdgcn_raw_buffer_load_b32(rsrc_s, s_offset(t, step), 0, 0);
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
  auto compute = [&](int stage, const i32x4(&breg)[NN], const u32x2(&sreg)[NN], int k, int h = 0) {
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
        a[mt] = a_stage[(mt * CHUNKS + c) * 16 + (lrow ^ (c & 7))];
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
          acc[mt][t] = mmac<T>(a0, b0, acc[mt][t]);
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
  // queue behind the outstanding B / A prefetch on vmcnt (5-14% of gemm2 with its NCH epilogues).
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
          packed[2 * t] = pack2<T>(lo[0], lo[1]);
          packed[2 * t + 1] = pack2<T>(hi[0], hi[1]);
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
  zero_acc();

  if constexpr (NCH > 1)
  {
    // ---- resident A (size_k == 2 * BK): A loaded once, NCH column chunks of BN per block ----
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
    for (int ch = 0; ch < NCH; ch++)
    {
      const bool more = ch + 1 < NCH;
      compute(0, b0, sc, 0);
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

} // namespace w4a16_prefill

// mode >= 500 entry: BM rows per block, WARPS warps (all along N), NN 16-column tiles per warp,
// NCH > 1: A resident in LDS (size_k == 256; other size_k fall back to NCH = 1) and NCH column chunks of WARPS * NN * 16 per block.
// GEMM2 multiplies by topk_weights (down projection).  WPE: minimum waves per SIMD the register
// allocation must allow (0 = compiler default); SCM: scale steps fetched per load (2 or 4).
// SILU (gemm1 only): fused silu_and_mul, output is [rows, size_n / 2] = silu(gate) * up.
// AH = 2: A staged in LDS per half K step (ST = 2, NCH = 1 only; BM128 fits two blocks per CU).
template <int BM, int WARPS, int NN, int NCH, int ST, bool PREFETCH_A, bool GEMM2, int WPE, int SCM, typename T, bool SILU,
          int AH>
void launch_moe_w4a16_prefill(const GemmParams_w4a16<T> &p)
{
  constexpr int BN = WARPS * NN * 16;
  TORCH_CHECK(p.size_k % (2 * w4a16_prefill::BK) == 0, "moe_w4a16 prefill: size_k=", p.size_k,
              " must be a multiple of ", 2 * w4a16_prefill::BK);
  if constexpr (NCH > 1)
  {
    // Resident A only fits when the whole reduction is two K steps (size_k == 256, i.e. N == 256
    // for gemm2); any other size_k streams A through the same tile with one column chunk per block.
    if (p.size_k != 2 * w4a16_prefill::BK)
    {
      launch_moe_w4a16_prefill<BM, WARPS, NN, 1, ST, PREFETCH_A, GEMM2, WPE, SCM, T, SILU, AH>(p);
      return;
    }
  }
  TORCH_CHECK(p.size_n % (BN * NCH) == 0, "moe_w4a16 prefill: size_n=", p.size_n, " must be a multiple of ", BN * NCH);
  static_assert(NN % 2 == 0, "NN must be even (16 byte stores)");

  static_assert(ST == 1 || ST == 2, "LDS stages");
  static_assert(NCH == 1 || ST == 2, "resident-A mode keeps both K halves in LDS");
  static_assert(!SILU || (!GEMM2 && NCH == 1), "fused silu_and_mul is a gemm1 streamed-A epilogue");
  const int lds_bytes = ST * BM * w4a16_prefill::BK * (int)sizeof(T) / AH;
  dim3 grid(p.size_n / (BN * NCH), DIVIDE(p.sorted_token_lens, BM), 1);
  dim3 block(WARPS * 64, 1, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  if constexpr (GEMM2)
  {
    TORCH_CHECK(p.topk_weights != nullptr, "moe_w4a16 prefill gemm2 needs topk_weights");
    w4a16_prefill::moe_w4a16_prefill_kernel<T, BM, WARPS, NN, NCH, ST, true, PREFETCH_A, WPE, SCM, false, AH><<<grid, block, lds_bytes, stream>>>(
        p.ptr_A, p.ptr_B0, p.ptr_C, p.ptr_B_scale, p.topk_weights, p.sorted_token_ids, p.expert_ids,
        p.num_tokens_post_pad_ptr, p.size_m, p.size_n, p.size_k, p.sorted_token_lens, p.top_k);
  }
  else
  {
    w4a16_prefill::moe_w4a16_prefill_kernel<T, BM, WARPS, NN, NCH, ST, false, PREFETCH_A, WPE, SCM, SILU, AH><<<grid, block, lds_bytes, stream>>>(
        p.ptr_A, p.ptr_B0, p.ptr_C, p.ptr_B_scale, nullptr, p.sorted_token_ids, p.expert_ids,
        p.num_tokens_post_pad_ptr, p.size_m, p.size_n, p.size_k, p.sorted_token_lens, p.top_k);
  }
}
