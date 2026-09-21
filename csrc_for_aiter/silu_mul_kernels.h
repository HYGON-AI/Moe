// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <climits>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <type_traits>

#include "compat.h"
#include "dispatch_utils.h"
#include "intrinsic_2.h"

namespace moe_c
{

#ifdef USE_ROCM
  constexpr int kWarpSize = 64;
#else
  constexpr int kWarpSize = 32;
#endif
  constexpr float kLog2e = 1.44269504088896340736f;

  template <typename scalar_t>
  using native_t = typename std::conditional_t<
      std::is_same_v<scalar_t, c10::Half>, __half,
      typename std::conditional_t<std::is_same_v<scalar_t, c10::BFloat16>,
                                  __hip_bfloat16, scalar_t>>;

  __device__ __forceinline__ float silu_fast_sigmoid(float x)
  {
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
    return __builtin_amdgcn_rcpf(
        1.0f + __builtin_amdgcn_exp2f(-x * kLog2e));
#else
    return 1.0f / (1.0f + expf(-x));
#endif
  }

  __device__ __forceinline__ float silu_fast(float x)
  {
    return x * silu_fast_sigmoid(x);
  }

  __device__ __forceinline__ float silu_fast_tanh(float x);

  __device__ __forceinline__ float gelu_exact(float x)
  {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
  }

  __device__ __forceinline__ float gelu_tanh_approx(float x)
  {
    constexpr float kBeta = 0.79788456080286535588f;
    constexpr float kKappa = 0.044715f;
    const float x2 = x * x;
    const float inner = kBeta * x * (1.0f + kKappa * x2);
    return 0.5f * x * (1.0f + silu_fast_tanh(inner));
  }

  __device__ __forceinline__ uint32_t gelu_tanh_half2_packed(uint32_t packed)
  {
    const __half2 x = *reinterpret_cast<const __half2 *>(&packed);
    const __half2 one = __float2half2_rn(1.0f);
    const __half2 scale = __float2half2_rn(-1.5957691216057308f);
    const __half2 cubic = __float2half2_rn(0.044715f);
    const __half2 x2 = __hmul2(x, x);
    const __half2 exponent =
        __hmul2(__hmul2(scale, x), __hadd2(one, __hmul2(cubic, x2)));
    const __half2 denominator = __hadd2(one, h2exp(exponent));
    const __half2 result = __hmul2(x, h2rcp(denominator));
    return *reinterpret_cast<const uint32_t *>(&result);
  }

  __device__ __forceinline__ uint32_t silu_pack_bf16_pair(float lo, float hi)
  {
#if defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__)
    const __hip_bfloat16 lo_b = b32_to_b16<__hip_bfloat16>(lo);
    const __hip_bfloat16 hi_b = b32_to_b16<__hip_bfloat16>(hi);
    uint32_t packed = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo_b));
    packed |= static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi_b)) << 16;
    return packed;
#elif defined(__gfx938__)
    __builtin_amdgcn_sched_barrier(0);
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2\n\t"
                 : "=v"(packed)
                 : "v"(lo), "v"(hi));
    __builtin_amdgcn_sched_barrier(0);
    return packed;
#else
    const __hip_bfloat16 lo_b = b32_to_b16<__hip_bfloat16>(lo);
    const __hip_bfloat16 hi_b = b32_to_b16<__hip_bfloat16>(hi);
    uint32_t packed = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo_b));
    packed |= static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi_b)) << 16;
    return packed;
#endif
  }

  __device__ __forceinline__ uint32_t silu_pack_fp16_pair(float lo, float hi)
  {
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
    const auto packed_h2 = __builtin_amdgcn_cvt_pkrtz(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&packed_h2);
#else
    const __half lo_h = b32_to_b16<__half>(lo);
    const __half hi_h = b32_to_b16<__half>(hi);
    uint32_t packed = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo_h));
    packed |= static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi_h)) << 16;
    return packed;
#endif
  }

  __device__ __forceinline__ float silu_fast_tanh(float x)
  {
    const float ax = fabsf(x);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
    const float t = __builtin_amdgcn_exp2f(-2.0f * ax * kLog2e);
    const float y = (1.0f - t) * __builtin_amdgcn_rcpf(1.0f + t);
#else
    const float t = exp2f(-2.0f * ax * kLog2e);
    const float y = (1.0f - t) / (1.0f + t);
#endif
    return copysignf(y, x);
  }

  __device__ __forceinline__ float situ_gate_part(float gate_f, float beta1)
  {
    return beta1 * silu_fast_tanh(gate_f / beta1) * silu_fast_sigmoid(gate_f);
  }

  __device__ __forceinline__ float situ_up_part(float up_f, float beta2)
  {
    return beta2 * silu_fast_tanh(up_f / beta2);
  }

  __device__ __forceinline__ float situ_gate_part_beta4(float gate_f)
  {
    // Folded identity:  gate(g) = 4*tanh(g/4)*sigmoid(g)
    //                 = 4*sign(g) * (1-b) * s(g)  ,  b = exp(-|g|/2)
    //   where s(g) = 1/((1+b)*(1+b^2))  for g>=0
    //         s(g) = b^2 * 1/((1+b)*(1+b^2))  for g<0
    // Cuts the two reciprocals of the naive form (one for tanh, one for
    // sigmoid) down to a single rcp over the fused denominator; 
    // numerically equivalent with the two-rcp implementation.
    const float ax = fabsf(gate_f);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
    const float b = __builtin_amdgcn_exp2f(-0.7213475204444817f * ax);
    const float a = b * b;
    const float rcp_den = __builtin_amdgcn_rcpf((1.0f + b) * (1.0f + a));
#else
    const float b = exp2f(-0.7213475204444817f * ax);
    const float a = b * b;
    const float rcp_den = 1.0f / ((1.0f + b) * (1.0f + a));
#endif
    const float num = (gate_f >= 0.0f) ? (1.0f - b) : (1.0f - b) * a;
    return 4.0f * copysignf(num * rcp_den, gate_f);
  }

  __device__ __forceinline__ float situ_up_part_beta25(float up_f)
  {
    const float ax = fabsf(up_f);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
    const float t = __builtin_amdgcn_exp2f(-0.11541560327111707f * ax);
    const float y = (1.0f - t) * __builtin_amdgcn_rcpf(1.0f + t);
#else
    const float t = exp2f(-0.11541560327111707f * ax);
    const float y = (1.0f - t) / (1.0f + t);
#endif
    return 25.0f * copysignf(y, up_f);
  }

  template <int ACTIVATION>
  __device__ __forceinline__ float glu_activation_elem(
      float gate_f, float up_f, float beta1, float beta2)
  {
    if constexpr (ACTIVATION == 0)
    {
      return silu_fast(gate_f) * up_f;
    }
    else if constexpr (ACTIVATION == 2)
    {
      return situ_gate_part_beta4(gate_f) * situ_up_part_beta25(up_f);
    }
    else if constexpr (ACTIVATION == 3)
    {
      return gelu_exact(gate_f) * up_f;
    }
    else if constexpr (ACTIVATION == 4)
    {
      return gelu_tanh_approx(gate_f) * up_f;
    }
    else
    {
      return situ_gate_part(gate_f, beta1) * situ_up_part(up_f, beta2);
    }
  }

  template <int ACTIVATION>
  __device__ __forceinline__ float activation_elem(float x)
  {
    if constexpr (ACTIVATION == 3)
    {
      return gelu_exact(x);
    }
    else if constexpr (ACTIVATION == 4)
    {
      return gelu_tanh_approx(x);
    }
    else
    {
      return x;
    }
  }

  __device__ __forceinline__ int silu_readfirstlane(int x)
  {
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
    return __builtin_amdgcn_readfirstlane(x);
#else
    return x;
#endif
  }

  template <int kBytes, bool NT>
  __device__ __forceinline__ void silu_gmem_load(void *dst, const void *src)
  {
    if constexpr (kBytes == 4)
    {
      auto *d = reinterpret_cast<uint32_t *>(dst);
      auto *s = reinterpret_cast<const uint32_t *>(src);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
      *d = NT ? __builtin_nontemporal_load(s) : *s;
#else
      *d = *s;
#endif
    }
    else if constexpr (kBytes == 8)
    {
      auto *d = reinterpret_cast<uint64_t *>(dst);
      auto *s = reinterpret_cast<const uint64_t *>(src);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
      *d = NT ? __builtin_nontemporal_load(s) : *s;
#else
      *d = *s;
#endif
    }
    else if constexpr (kBytes == 16)
    {
      auto *d = reinterpret_cast<uint64_t *>(dst);
      auto *s = reinterpret_cast<const uint64_t *>(src);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
      if constexpr (NT)
      {
        d[0] = __builtin_nontemporal_load(s);
        d[1] = __builtin_nontemporal_load(s + 1);
      }
      else
#endif
      {
        d[0] = s[0];
        d[1] = s[1];
      }
    }
  }

  template <int kBytes, bool NT>
  __device__ __forceinline__ void silu_gmem_store(void *dst, const void *src)
  {
    if constexpr (kBytes == 4)
    {
      auto *d = reinterpret_cast<uint32_t *>(dst);
      auto *s = reinterpret_cast<const uint32_t *>(src);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
      if constexpr (NT)
      {
        __builtin_nontemporal_store(*s, d);
      }
      else
#endif
      {
        *d = *s;
      }
    }
    else if constexpr (kBytes == 8)
    {
      auto *d = reinterpret_cast<uint64_t *>(dst);
      auto *s = reinterpret_cast<const uint64_t *>(src);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
      if constexpr (NT)
      {
        __builtin_nontemporal_store(*s, d);
      }
      else
#endif
      {
        *d = *s;
      }
    }
    else if constexpr (kBytes == 16)
    {
      auto *d = reinterpret_cast<uint64_t *>(dst);
      auto *s = reinterpret_cast<const uint64_t *>(src);
#if defined(USE_ROCM) || defined(__HIPCC__) || defined(__DTK_ARCH__)
      if constexpr (NT)
      {
        __builtin_nontemporal_store(s[0], d);
        __builtin_nontemporal_store(s[1], d + 1);
      }
      else
#endif
      {
        d[0] = s[0];
        d[1] = s[1];
      }
    }
  }

  template <int kDwords, bool NT>
  __device__ __forceinline__ void silu_load_gate_up(
      uint32_t *gate_buf, uint32_t *up_buf,
      const uint32_t *gate_dw, const uint32_t *up_dw)
  {
    if constexpr (kDwords == 1)
    {
      silu_gmem_load<4, NT>(gate_buf, gate_dw);
      silu_gmem_load<4, NT>(up_buf, up_dw);
    }
    else if constexpr (kDwords == 2)
    {
      silu_gmem_load<8, NT>(gate_buf, gate_dw);
      silu_gmem_load<8, NT>(up_buf, up_dw);
    }
    else if constexpr (kDwords == 4)
    {
      silu_gmem_load<16, NT>(gate_buf, gate_dw);
      silu_gmem_load<16, NT>(up_buf, up_dw);
    }
    else
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        silu_gmem_load<4, NT>(&gate_buf[d], &gate_dw[d]);
        silu_gmem_load<4, NT>(&up_buf[d], &up_dw[d]);
      }
    }
  }

  template <int kDwords>
  __device__ __forceinline__ void silu_store_out(
      uint32_t *out_dw, const uint32_t *out_vals)
  {
    if constexpr (kDwords == 1)
    {
      silu_gmem_store<4, false>(out_dw, out_vals);
    }
    else if constexpr (kDwords == 2)
    {
      silu_gmem_store<8, false>(out_dw, out_vals);
    }
    else if constexpr (kDwords == 4)
    {
      silu_gmem_store<16, false>(out_dw, out_vals);
    }
    else
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        silu_gmem_store<4, false>(&out_dw[d], &out_vals[d]);
      }
    }
  }

  template <int kDwords, bool NT>
  __device__ __forceinline__ void activation_load_input(
      uint32_t *in_buf, const uint32_t *in_dw)
  {
    if constexpr (kDwords == 1)
    {
      silu_gmem_load<4, NT>(in_buf, in_dw);
    }
    else if constexpr (kDwords == 2)
    {
      silu_gmem_load<8, NT>(in_buf, in_dw);
    }
    else if constexpr (kDwords == 4)
    {
      silu_gmem_load<16, NT>(in_buf, in_dw);
    }
    else
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        silu_gmem_load<4, NT>(&in_buf[d], &in_dw[d]);
      }
    }
  }

  template <typename native_scalar_t, int VEC_SIZE, bool N_DIV, int ACTIVATION>
  __device__ __forceinline__ void glu_tile_scalar(
      const native_scalar_t *gate_ptr,
      const native_scalar_t *up_ptr,
      native_scalar_t *out_ptr,
      int n_base,
      int N,
      float beta1,
      float beta2)
  {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
    {
      if (n_base + i >= N)
      {
        return;
      }
      const float gate_f = static_cast<float>(gate_ptr[i]);
      const float up_f = static_cast<float>(up_ptr[i]);
      const float result_f =
          glu_activation_elem<ACTIVATION>(gate_f, up_f, beta1, beta2);
      if constexpr (std::is_same_v<native_scalar_t, float>)
      {
        out_ptr[i] = result_f;
      }
      else
      {
        out_ptr[i] = b32_to_b16<native_scalar_t>(result_f);
      }
    }
  }

  template <typename native_scalar_t, int VEC_SIZE, bool N_DIV>
  __device__ __forceinline__ void silu_and_mul_tile_scalar(
      const native_scalar_t *gate_ptr,
      const native_scalar_t *up_ptr,
      native_scalar_t *out_ptr,
      int n_base,
      int N)
  {
    glu_tile_scalar<native_scalar_t, VEC_SIZE, N_DIV, 0>(
        gate_ptr, up_ptr, out_ptr, n_base, N, 0.0f, 0.0f);
  }

  template <typename native_scalar_t, int VEC_SIZE, bool N_DIV, int ACTIVATION>
  __device__ __forceinline__ void glu_tile(
      const uint32_t *gate_dw,
      const uint32_t *up_dw,
      uint32_t *out_dw,
      int n_base,
      int N,
      float beta1,
      float beta2)
  {
    constexpr int kElemsPerDword = 4 / static_cast<int>(sizeof(native_scalar_t));

    if constexpr (VEC_SIZE % kElemsPerDword != 0)
    {
      glu_tile_scalar<native_scalar_t, VEC_SIZE, N_DIV, ACTIVATION>(
          reinterpret_cast<const native_scalar_t *>(gate_dw),
          reinterpret_cast<const native_scalar_t *>(up_dw),
          reinterpret_cast<native_scalar_t *>(out_dw), n_base, N, beta1, beta2);
      return;
    }

    constexpr int kDwords = VEC_SIZE / kElemsPerDword;

    uint32_t gate_buf[kDwords];
    uint32_t up_buf[kDwords];

    const bool full_vec = (n_base + VEC_SIZE <= N);

    if constexpr (N_DIV)
    {
      silu_load_gate_up<kDwords, true>(gate_buf, up_buf, gate_dw, up_dw);
    }
    else if (full_vec)
    {
      silu_load_gate_up<kDwords, true>(gate_buf, up_buf, gate_dw, up_dw);
    }
    else
    {
      if (n_base >= N)
      {
        return;
      }
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        const int elem = n_base + d * kElemsPerDword;
        if (elem + kElemsPerDword <= N)
        {
          silu_gmem_load<4, false>(&gate_buf[d], &gate_dw[d]);
          silu_gmem_load<4, false>(&up_buf[d], &up_dw[d]);
        }
        else
        {
          gate_buf[d] = 0;
          up_buf[d] = 0;
          const native_scalar_t *gate_ptr =
              reinterpret_cast<const native_scalar_t *>(gate_dw) + d * kElemsPerDword;
          const native_scalar_t *up_ptr =
              reinterpret_cast<const native_scalar_t *>(up_dw) + d * kElemsPerDword;
          native_scalar_t *gate_out =
              reinterpret_cast<native_scalar_t *>(&gate_buf[d]);
          native_scalar_t *up_out =
              reinterpret_cast<native_scalar_t *>(&up_buf[d]);
#pragma unroll
          for (int i = 0; i < kElemsPerDword; ++i)
          {
            if (elem + i < N)
            {
              gate_out[i] = gate_ptr[i];
              up_out[i] = up_ptr[i];
            }
          }
        }
      }
    }

    const native_scalar_t *gate_vals =
        reinterpret_cast<const native_scalar_t *>(gate_buf);
    const native_scalar_t *up_vals =
        reinterpret_cast<const native_scalar_t *>(up_buf);

    uint32_t out_packed[kDwords];
    native_scalar_t out_tail[VEC_SIZE];

    if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16>)
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        const int i = d * kElemsPerDword;
        const float gate_f0 = static_cast<float>(gate_vals[i]);
        const float up_f0 = static_cast<float>(up_vals[i]);
        const float result_f0 =
            glu_activation_elem<ACTIVATION>(gate_f0, up_f0, beta1, beta2);
        const float gate_f1 = static_cast<float>(gate_vals[i + 1]);
        const float up_f1 = static_cast<float>(up_vals[i + 1]);
        const float result_f1 =
            glu_activation_elem<ACTIVATION>(gate_f1, up_f1, beta1, beta2);
        out_packed[d] = silu_pack_bf16_pair(result_f0, result_f1);
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i)
      {
        const float gate_f = static_cast<float>(gate_vals[i]);
        const float up_f = static_cast<float>(up_vals[i]);
        const float result_f =
            glu_activation_elem<ACTIVATION>(gate_f, up_f, beta1, beta2);
        if constexpr (std::is_same_v<native_scalar_t, float>)
        {
          out_tail[i] = result_f;
        }
        else
        {
          out_tail[i] = b32_to_b16<native_scalar_t>(result_f);
        }
      }
    }

    if constexpr (N_DIV)
    {
      if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16>)
      {
        silu_store_out<kDwords>(out_dw, out_packed);
      }
      else
      {
        silu_store_out<kDwords>(out_dw, reinterpret_cast<const uint32_t *>(out_tail));
      }
    }
    else if (full_vec)
    {
      if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16>)
      {
        silu_store_out<kDwords>(out_dw, out_packed);
      }
      else
      {
        silu_store_out<kDwords>(out_dw, reinterpret_cast<const uint32_t *>(out_tail));
      }
    }
    else
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        const int elem = n_base + d * kElemsPerDword;
        if (elem + kElemsPerDword <= N)
        {
          if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16>)
          {
            silu_gmem_store<4, false>(&out_dw[d], &out_packed[d]);
          }
          else
          {
            silu_gmem_store<4, false>(
                &out_dw[d],
                &reinterpret_cast<const uint32_t *>(out_tail)[d]);
          }
        }
        else
        {
          native_scalar_t *out_ptr =
              reinterpret_cast<native_scalar_t *>(out_dw) + d * kElemsPerDword;
#pragma unroll
          for (int i = 0; i < kElemsPerDword; ++i)
          {
            if (elem + i < N)
            {
              const int out_i = d * kElemsPerDword + i;
              if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16>)
              {
                out_ptr[i] = reinterpret_cast<const native_scalar_t *>(out_packed)[out_i];
              }
              else
              {
                out_ptr[i] = out_tail[out_i];
              }
            }
          }
        }
      }
    }
  }

  template <typename native_scalar_t, int VEC_SIZE, bool N_DIV>
  __device__ __forceinline__ void silu_and_mul_tile(
      const uint32_t *gate_dw,
      const uint32_t *up_dw,
      uint32_t *out_dw,
      int n_base,
      int N)
  {
    glu_tile<native_scalar_t, VEC_SIZE, N_DIV, 0>(
        gate_dw, up_dw, out_dw, n_base, N, 0.0f, 0.0f);
  }

  template <typename native_scalar_t, int VEC_SIZE, bool N_DIV, int ACTIVATION>
  __device__ __forceinline__ void activation_tile_scalar(
      const native_scalar_t *in_ptr,
      native_scalar_t *out_ptr,
      int n_base,
      int N)
  {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
    {
      if (n_base + i >= N)
      {
        return;
      }
      const float x = static_cast<float>(in_ptr[i]);
      const float result_f = activation_elem<ACTIVATION>(x);
      if constexpr (std::is_same_v<native_scalar_t, float>)
      {
        out_ptr[i] = result_f;
      }
      else
      {
        out_ptr[i] = b32_to_b16<native_scalar_t>(result_f);
      }
    }
  }

  template <typename native_scalar_t, int VEC_SIZE, bool N_DIV, int ACTIVATION>
  __device__ __forceinline__ void activation_tile(
      const uint32_t *in_dw,
      uint32_t *out_dw,
      int n_base,
      int N)
  {
    constexpr int kElemsPerDword = 4 / static_cast<int>(sizeof(native_scalar_t));

    if constexpr (VEC_SIZE % kElemsPerDword != 0)
    {
      activation_tile_scalar<native_scalar_t, VEC_SIZE, N_DIV, ACTIVATION>(
          reinterpret_cast<const native_scalar_t *>(in_dw),
          reinterpret_cast<native_scalar_t *>(out_dw), n_base, N);
      return;
    }

    constexpr int kDwords = VEC_SIZE / kElemsPerDword;
    uint32_t in_buf[kDwords];

    const bool full_vec = (n_base + VEC_SIZE <= N);

    if constexpr (N_DIV)
    {
      activation_load_input<kDwords, true>(in_buf, in_dw);
    }
    else if (full_vec)
    {
      activation_load_input<kDwords, true>(in_buf, in_dw);
    }
    else
    {
      if (n_base >= N)
      {
        return;
      }
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        const int elem = n_base + d * kElemsPerDword;
        if (elem + kElemsPerDword <= N)
        {
          silu_gmem_load<4, false>(&in_buf[d], &in_dw[d]);
        }
        else
        {
          in_buf[d] = 0;
          const native_scalar_t *in_ptr =
              reinterpret_cast<const native_scalar_t *>(in_dw) + d * kElemsPerDword;
          native_scalar_t *in_out =
              reinterpret_cast<native_scalar_t *>(&in_buf[d]);
#pragma unroll
          for (int i = 0; i < kElemsPerDword; ++i)
          {
            if (elem + i < N)
            {
              in_out[i] = in_ptr[i];
            }
          }
        }
      }
    }

    const native_scalar_t *in_vals =
        reinterpret_cast<const native_scalar_t *>(in_buf);

    uint32_t out_packed[kDwords];
    native_scalar_t out_tail[VEC_SIZE];

    if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16>)
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        const int i = d * kElemsPerDword;
        const float x0 = static_cast<float>(in_vals[i]);
        const float result_f0 = activation_elem<ACTIVATION>(x0);
        const float x1 = static_cast<float>(in_vals[i + 1]);
        const float result_f1 = activation_elem<ACTIVATION>(x1);
        out_packed[d] = silu_pack_bf16_pair(result_f0, result_f1);
      }
    }
    else if constexpr (std::is_same_v<native_scalar_t, __half> &&
                       ACTIVATION == 4)
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        out_packed[d] = gelu_tanh_half2_packed(in_buf[d]);
      }
    }
    else if constexpr (std::is_same_v<native_scalar_t, __half>)
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        const int i = d * kElemsPerDword;
        const float x0 = static_cast<float>(in_vals[i]);
        const float result_f0 = activation_elem<ACTIVATION>(x0);
        const float x1 = static_cast<float>(in_vals[i + 1]);
        const float result_f1 = activation_elem<ACTIVATION>(x1);
        out_packed[d] = silu_pack_fp16_pair(result_f0, result_f1);
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i)
      {
        const float x = static_cast<float>(in_vals[i]);
        const float result_f = activation_elem<ACTIVATION>(x);
        if constexpr (std::is_same_v<native_scalar_t, float>)
        {
          out_tail[i] = result_f;
        }
        else
        {
          out_tail[i] = b32_to_b16<native_scalar_t>(result_f);
        }
      }
    }

    if constexpr (N_DIV)
    {
      if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16> ||
                    std::is_same_v<native_scalar_t, __half>)
      {
        silu_store_out<kDwords>(out_dw, out_packed);
      }
      else
      {
        silu_store_out<kDwords>(out_dw, reinterpret_cast<const uint32_t *>(out_tail));
      }
    }
    else if (full_vec)
    {
      if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16> ||
                    std::is_same_v<native_scalar_t, __half>)
      {
        silu_store_out<kDwords>(out_dw, out_packed);
      }
      else
      {
        silu_store_out<kDwords>(out_dw, reinterpret_cast<const uint32_t *>(out_tail));
      }
    }
    else
    {
#pragma unroll
      for (int d = 0; d < kDwords; ++d)
      {
        const int elem = n_base + d * kElemsPerDword;
        if (elem + kElemsPerDword <= N)
        {
          if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16> ||
                        std::is_same_v<native_scalar_t, __half>)
          {
            silu_gmem_store<4, false>(&out_dw[d], &out_packed[d]);
          }
          else
          {
            silu_gmem_store<4, false>(
                &out_dw[d],
                &reinterpret_cast<const uint32_t *>(out_tail)[d]);
          }
        }
        else
        {
          native_scalar_t *out_ptr =
              reinterpret_cast<native_scalar_t *>(out_dw) + d * kElemsPerDword;
#pragma unroll
          for (int i = 0; i < kElemsPerDword; ++i)
          {
            if (elem + i < N)
            {
              const int out_i = d * kElemsPerDword + i;
              if constexpr (std::is_same_v<native_scalar_t, __hip_bfloat16> ||
                            std::is_same_v<native_scalar_t, __half>)
              {
                out_ptr[i] = reinterpret_cast<const native_scalar_t *>(out_packed)[out_i];
              }
              else
              {
                out_ptr[i] = out_tail[out_i];
              }
            }
          }
        }
      }
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int VEC_SIZE, int ROWS_PER_BLOCK,
            bool N_DIV, int ACTIVATION>
  __launch_bounds__(BLOCK_SIZE_N / VEC_SIZE,
                    (BLOCK_SIZE_N / VEC_SIZE <= 128 ? 2 : 1)) __global__
      void activation_kernel(
          scalar_t *__restrict__ out,         // [M, N]
          const scalar_t *__restrict__ input, // [M, N]
          int M, int N)
  {
    using native_scalar_t = native_t<scalar_t>;
    constexpr int kElemsPerDword = 4 / static_cast<int>(sizeof(native_scalar_t));

    const int pid = blockIdx.x;
    const int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    const int pid_m_group = pid / num_pid_n;
    const int pid_n = pid - pid_m_group * num_pid_n;

    const int n_base = pid_n * BLOCK_SIZE_N + threadIdx.x * VEC_SIZE;
    if constexpr (!N_DIV)
    {
      if (n_base >= N)
      {
        return;
      }
    }
    const int dw_base = n_base / kElemsPerDword;

    const int m_base = pid_m_group * ROWS_PER_BLOCK;

#pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r)
    {
      const int pid_m = silu_readfirstlane(m_base + r);

      const int64_t row_off = static_cast<int64_t>(pid_m) * N;
      const uint32_t *in_dw =
          reinterpret_cast<const uint32_t *>(input + row_off) + dw_base;
      uint32_t *out_dw =
          reinterpret_cast<uint32_t *>(out + row_off) + dw_base;

      if constexpr (VEC_SIZE % kElemsPerDword != 0)
      {
        const native_scalar_t *in_ptr =
            reinterpret_cast<const native_scalar_t *>(input + row_off) + n_base;
        native_scalar_t *out_ptr =
            reinterpret_cast<native_scalar_t *>(out + row_off) + n_base;

        activation_tile_scalar<native_scalar_t, VEC_SIZE, N_DIV, ACTIVATION>(
            in_ptr, out_ptr, n_base, N);
      }
      else
      {
        activation_tile<native_scalar_t, VEC_SIZE, N_DIV, ACTIVATION>(
            in_dw, out_dw, n_base, N);
      }
    }
  }

  // Each block handles ROWS_PER_BLOCK consecutive M rows on the same N-tile.
  template <typename scalar_t, int BLOCK_SIZE_N, int VEC_SIZE, int ROWS_PER_BLOCK,
            bool N_DIV, bool M_DIV, int ACTIVATION>
  __launch_bounds__(BLOCK_SIZE_N / VEC_SIZE,
                    (BLOCK_SIZE_N / VEC_SIZE <= 128 ? 2 : 1)) __global__
      void glu_kernel(
          scalar_t *__restrict__ out,         // [M, N]
          const scalar_t *__restrict__ input, // [M, 2N]
          int M, int N, float beta1, float beta2)
  {
    using native_scalar_t = native_t<scalar_t>;
    constexpr int kElemsPerDword = 4 / static_cast<int>(sizeof(native_scalar_t));

    const int pid = blockIdx.x;
    const int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    const int pid_m_group = pid / num_pid_n;
    const int pid_n = pid - pid_m_group * num_pid_n;

    const int n_base = pid_n * BLOCK_SIZE_N + threadIdx.x * VEC_SIZE;
    if constexpr (!N_DIV)
    {
      if (n_base >= N)
      {
        return;
      }
    }
    const int dw_base = n_base / kElemsPerDword;

    const int m_base = pid_m_group * ROWS_PER_BLOCK;
    if constexpr (!M_DIV)
    {
      if (m_base >= M)
      {
        return;
      }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r)
    {
      const int pid_m = silu_readfirstlane(m_base + r);
      if constexpr (!M_DIV)
      {
        if (pid_m >= M)
        {
          continue;
        }
      }

      const int64_t in_row_off = static_cast<int64_t>(pid_m) * (int64_t{2} * N);
      const int64_t out_row_off = static_cast<int64_t>(pid_m) * N;

      const uint32_t *gate_dw =
          reinterpret_cast<const uint32_t *>(input + in_row_off) + dw_base;
      const uint32_t *up_dw =
          reinterpret_cast<const uint32_t *>(input + in_row_off + N) + dw_base;
      uint32_t *out_dw =
          reinterpret_cast<uint32_t *>(out + out_row_off) + dw_base;

      if constexpr (VEC_SIZE % kElemsPerDword != 0)
      {
        const native_scalar_t *gate_ptr =
            reinterpret_cast<const native_scalar_t *>(input + in_row_off) + n_base;
        const native_scalar_t *up_ptr =
            reinterpret_cast<const native_scalar_t *>(input + in_row_off + N) + n_base;
        native_scalar_t *out_ptr =
            reinterpret_cast<native_scalar_t *>(out + out_row_off) + n_base;

        glu_tile_scalar<native_scalar_t, VEC_SIZE, N_DIV, ACTIVATION>(
            gate_ptr, up_ptr, out_ptr, n_base, N, beta1, beta2);
      }
      else
      {
        glu_tile<native_scalar_t, VEC_SIZE, N_DIV, ACTIVATION>(
            gate_dw, up_dw, out_dw, n_base, N, beta1, beta2);
      }
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int VEC_SIZE, int ROWS_PER_BLOCK,
            bool N_DIV, bool M_DIV, int ACTIVATION>
  inline void launch_glu_kernel(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N, cudaStream_t stream,
      int grid, int kThreads, float beta1, float beta2)
  {
    glu_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, ROWS_PER_BLOCK, N_DIV, M_DIV,
               ACTIVATION>
        <<<grid, kThreads, 0, stream>>>(out_ptr, in_ptr, M, N, beta1, beta2);
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int VEC_SIZE, int ROWS_PER_BLOCK,
            bool N_DIV, int ACTIVATION>
  inline void launch_activation_kernel(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N, cudaStream_t stream,
      int grid, int kThreads)
  {
    activation_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, ROWS_PER_BLOCK, N_DIV,
                      ACTIVATION>
        <<<grid, kThreads, 0, stream>>>(out_ptr, in_ptr, M, N);
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int VEC_SIZE, int ROWS_PER_BLOCK,
            int ACTIVATION>
  inline void launch_glu(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N, cudaStream_t stream,
      float beta1, float beta2)
  {
    constexpr int kThreads = BLOCK_SIZE_N / VEC_SIZE;
    static_assert(BLOCK_SIZE_N % VEC_SIZE == 0,
                  "BLOCK_SIZE_N must be divisible by VEC_SIZE");

    const int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    const int num_pid_m = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int grid = num_pid_m * num_pid_n;
    const bool n_div = (N % BLOCK_SIZE_N) == 0;
    const bool m_div = (M % ROWS_PER_BLOCK) == 0;

    if (n_div && m_div)
    {
      launch_glu_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, ROWS_PER_BLOCK,
                        true, true, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, grid, kThreads, beta1, beta2);
    }
    else
    {
      launch_glu_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, ROWS_PER_BLOCK,
                        false, false, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, grid, kThreads, beta1, beta2);
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int VEC_SIZE, int ROWS_PER_BLOCK,
            int ACTIVATION>
  inline void launch_activation(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N,
      cudaStream_t stream)
  {
    constexpr int kThreads = BLOCK_SIZE_N / VEC_SIZE;
    static_assert(BLOCK_SIZE_N % VEC_SIZE == 0,
                  "BLOCK_SIZE_N must be divisible by VEC_SIZE");

    const int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    const bool n_div = (N % BLOCK_SIZE_N) == 0;
    const int main_m = (M / ROWS_PER_BLOCK) * ROWS_PER_BLOCK;

    if (main_m > 0)
    {
      const int main_grid = (main_m / ROWS_PER_BLOCK) * num_pid_n;
      if (n_div)
      {
        launch_activation_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, ROWS_PER_BLOCK,
                                 true, ACTIVATION>(
            out_ptr, in_ptr, main_m, N, stream, main_grid, kThreads);
      }
      else
      {
        launch_activation_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, ROWS_PER_BLOCK,
                                 false, ACTIVATION>(
            out_ptr, in_ptr, main_m, N, stream, main_grid, kThreads);
      }
    }

    const int tail_m = M - main_m;
    if (tail_m > 0)
    {
      const int tail_grid = tail_m * num_pid_n;
      scalar_t *tail_out_ptr = out_ptr + static_cast<int64_t>(main_m) * N;
      const scalar_t *tail_in_ptr = in_ptr + static_cast<int64_t>(main_m) * N;
      if (n_div)
      {
        launch_activation_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, 1,
                                 true, ACTIVATION>(
            tail_out_ptr, tail_in_ptr, tail_m, N, stream, tail_grid, kThreads);
      }
      else
      {
        launch_activation_kernel<scalar_t, BLOCK_SIZE_N, VEC_SIZE, 1,
                                 false, ACTIVATION>(
            tail_out_ptr, tail_in_ptr, tail_m, N, stream, tail_grid, kThreads);
      }
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int ROWS_PER_BLOCK,
            int ACTIVATION>
  inline void launch_glu_vec(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N, cudaStream_t stream,
      int vec, float beta1, float beta2)
  {
    if (vec == 4)
    {
      launch_glu<scalar_t, BLOCK_SIZE_N, 4, ROWS_PER_BLOCK, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, beta1, beta2);
    }
    else if (vec == 2)
    {
      launch_glu<scalar_t, BLOCK_SIZE_N, 2, ROWS_PER_BLOCK, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, beta1, beta2);
    }
    else
    {
      launch_glu<scalar_t, BLOCK_SIZE_N, 1, ROWS_PER_BLOCK, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, beta1, beta2);
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int ROWS_PER_BLOCK,
            int ACTIVATION>
  inline void launch_activation_vec(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N,
      cudaStream_t stream, int vec)
  {
    if (vec == 4)
    {
      launch_activation<scalar_t, BLOCK_SIZE_N, 4, ROWS_PER_BLOCK, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream);
    }
    else if (vec == 2)
    {
      launch_activation<scalar_t, BLOCK_SIZE_N, 2, ROWS_PER_BLOCK, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream);
    }
    else
    {
      launch_activation<scalar_t, BLOCK_SIZE_N, 1, ROWS_PER_BLOCK, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream);
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int ACTIVATION>
  inline void launch_glu_rows(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N, cudaStream_t stream,
      int vec, int rows_per_block, float beta1, float beta2)
  {
    switch (rows_per_block)
    {
    case 1:
      launch_glu_vec<scalar_t, BLOCK_SIZE_N, 1, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec, beta1, beta2);
      break;
    case 2:
      launch_glu_vec<scalar_t, BLOCK_SIZE_N, 2, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec, beta1, beta2);
      break;
    case 4:
      launch_glu_vec<scalar_t, BLOCK_SIZE_N, 4, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec, beta1, beta2);
      break;
    case 8:
      launch_glu_vec<scalar_t, BLOCK_SIZE_N, 8, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec, beta1, beta2);
      break;
    default:
      TORCH_CHECK(false,
                  "silu_and_mul: unsupported rows_per_block=", rows_per_block,
                  " (supported: 1, 2, 4, 8)");
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N, int ACTIVATION>
  inline void launch_activation_rows(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N,
      cudaStream_t stream, int vec, int rows_per_block)
  {
    switch (rows_per_block)
    {
    case 1:
      launch_activation_vec<scalar_t, BLOCK_SIZE_N, 1, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec);
      break;
    case 2:
      launch_activation_vec<scalar_t, BLOCK_SIZE_N, 2, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec);
      break;
    case 4:
      launch_activation_vec<scalar_t, BLOCK_SIZE_N, 4, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec);
      break;
    case 8:
      launch_activation_vec<scalar_t, BLOCK_SIZE_N, 8, ACTIVATION>(
          out_ptr, in_ptr, M, N, stream, vec);
      break;
    default:
      TORCH_CHECK(false,
                  "activation: unsupported rows_per_block=", rows_per_block,
                  " (supported: 1, 2, 4, 8)");
    }
  }

  template <typename scalar_t, int BLOCK_SIZE_N>
  inline void launch_silu_and_mul_rows(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N, cudaStream_t stream,
      int vec, int rows_per_block)
  {
    launch_glu_rows<scalar_t, BLOCK_SIZE_N, 0>(
        out_ptr, in_ptr, M, N, stream, vec, rows_per_block, 0.0f, 0.0f);
  }

  template <typename scalar_t, int BLOCK_SIZE_N>
  inline void launch_situ_glu_rows(
      scalar_t *out_ptr, const scalar_t *in_ptr, int M, int N, cudaStream_t stream,
      int vec, int rows_per_block, float beta1, float beta2)
  {
    launch_glu_rows<scalar_t, BLOCK_SIZE_N, 1>(
        out_ptr, in_ptr, M, N, stream, vec, rows_per_block, beta1, beta2);
  }

  struct SiluLaunchConfig
  {
    int block_n;
    int rows_per_block;
    int vec_size;
  };

  constexpr int kSupportedBlockN[] = {128, 256, 384, 512,
                                      768, 1024, 1536, 2048};

  inline bool is_supported_block_n(int block_n)
  {
    for (int bn : kSupportedBlockN)
    {
      if (bn == block_n)
      {
        return true;
      }
    }
    return false;
  }

  inline int count_n_tiles(int N, int block_n)
  {
    return (N + block_n - 1) / block_n;
  }

  inline int active_threads_for_n(int N, int vec_size)
  {
    return (N + vec_size - 1) / vec_size;
  }

  inline bool is_valid_silu_thread_config(int block_n, int vec_size)
  {
    if (block_n % vec_size != 0)
    {
      return false;
    }
    const int threads = block_n / vec_size;
    return threads >= kWarpSize && (threads % kWarpSize) == 0 && threads <= 1024;
  }

  inline bool is_exact_n_tile_fit(int N, int vec_size)
  {
    return is_valid_silu_thread_config(N, vec_size) && is_supported_block_n(N);
  }

  inline int score_silu_launch_pair(int N, int block_n, int vec_size, int M)
  {
    const int tiles = count_n_tiles(N, block_n);
    const int threads = block_n / vec_size;
    const int active = active_threads_for_n(N, vec_size);
    const int idle = (tiles == 1) ? (threads - active) : 0;
    const int padding = tiles * block_n - N;

    int score = 0;
    if (tiles == 1)
    {
      score += 1'000'000;
    }
    score += threads * 100;
    score -= idle * 500;
    score -= padding;
    score -= (tiles - 1) * 50'000;
    if (tiles == 1 && M >= 4096)
    {
      score += vec_size * 20'000;
    }
    if (tiles == 1 && block_n == N)
    {
      score += 50'000;
    }
    if (tiles == 1 && M >= 4096 && threads > 512)
    {
      score -= (threads - 512) * 40;
    }
    return score;
  }

  inline int pick_silu_rows_per_block(
      int M, int rows_per_block, int block_n, int N)
  {
    if (rows_per_block != 1)
    {
      return rows_per_block;
    }
    if (M <= 16)
    {
      if (M <= 2)
      {
        return 8;
      }
      if (M <= 4)
      {
        return 4;
      }
      return 2;
    }
    if (M >= 4096 && count_n_tiles(N, block_n) == 1)
    {
      if (M >= 65536)
      {
        return 8;
      }
      if (M >= 32768)
      {
        return 4;
      }
      if (M >= 16384)
      {
        return 2;
      }
      return 1;
    }
    if (M >= 64)
    {
      return 2;
    }
    return 1;
  }

  inline int pick_situ_rows_per_block(
      int M, int rows_per_block, int block_n, int N)
  {
    if (rows_per_block != 1)
    {
      return rows_per_block;
    }
    if (M <= 16)
    {
      if (M <= 2)
      {
        return 8;
      }
      if (M <= 4)
      {
        return 4;
      }
      return 2;
    }
    if (M >= 4096 && count_n_tiles(N, block_n) == 1)
    {
      if (N >= 1536 && M >= 32768)
      {
        return 4;
      }
      if (M >= 32768 && N <= 384)
      {
        return 8;
      }
      if (M >= 16384 && N <= 1024)
      {
        return 4;
      }
      return 1;
    }
    if (M >= 64)
    {
      return 2;
    }
    return 1;
  }

  template <int ACTIVATION>
  inline SiluLaunchConfig pick_glu_launch_config(
      int M, int N, int rows_per_block, int vec_size)
  {
    if (M <= 16)
    {
      SiluLaunchConfig cfg;
      cfg.block_n = 128;
      cfg.vec_size = (vec_size == 1) ? 1 : 2;
      if constexpr (ACTIVATION == 0 || ACTIVATION == 3 || ACTIVATION == 4)
      {
        cfg.rows_per_block =
            pick_silu_rows_per_block(M, rows_per_block, cfg.block_n, N);
      }
      else
      {
        cfg.rows_per_block =
            pick_situ_rows_per_block(M, rows_per_block, cfg.block_n, N);
      }
      return cfg;
    }

    SiluLaunchConfig cfg;
    cfg.block_n = 512;
    cfg.vec_size = (vec_size == 1) ? 1 : 2;
    int best_score = INT_MIN;

    constexpr int kBlockNCandidates[] = {128, 256, 384, 512,
                                         768, 1024, 1536, 2048};
    constexpr int kVecCandidates[] = {1, 2, 4};

    auto consider_config = [&](int bn, int vec)
    {
      if (vec_size == 1 && vec != 1)
      {
        return;
      }
      if (vec_size != 1 && vec == 1)
      {
        return;
      }
      if (!is_valid_silu_thread_config(bn, vec))
      {
        return;
      }
      if (!is_supported_block_n(bn) && bn != N)
      {
        return;
      }
      const int threads = bn / vec;
      const int min_threads = (vec == 4) ? 192 : 128;
      if (M >= 4096 && threads < min_threads)
      {
        return;
      }
      const int score = score_silu_launch_pair(N, bn, vec, M);
      if (score > best_score)
      {
        best_score = score;
        cfg.block_n = bn;
        cfg.vec_size = vec;
      }
    };

    if (is_exact_n_tile_fit(N, 2))
    {
      consider_config(N, 2);
    }
    if (is_exact_n_tile_fit(N, 4))
    {
      consider_config(N, 4);
    }

    for (int bn : kBlockNCandidates)
    {
      if (N > 2048 && bn < 2048)
      {
        continue;
      }
      else if (N > 1024 && bn < 1024)
      {
        continue;
      }
      for (int vec : kVecCandidates)
      {
        consider_config(bn, vec);
      }
    }

    if constexpr (ACTIVATION == 0 || ACTIVATION == 3 || ACTIVATION == 4)
    {
      cfg.rows_per_block =
          pick_silu_rows_per_block(M, rows_per_block, cfg.block_n, N);
    }
    else
    {
      cfg.rows_per_block =
          pick_situ_rows_per_block(M, rows_per_block, cfg.block_n, N);
    }
    return cfg;
  }

  inline SiluLaunchConfig pick_silu_launch_config(
      int M, int N, int rows_per_block, int vec_size)
  {
    return pick_glu_launch_config<0>(M, N, rows_per_block, vec_size);
  }

  inline SiluLaunchConfig pick_situ_launch_config(
      int M, int N, int rows_per_block, int vec_size)
  {
    return pick_glu_launch_config<1>(M, N, rows_per_block, vec_size);
  }

  enum class GeluTanhActivationArch
  {
    Other,
    Gfx936,
    Gfx938,
  };

  inline GeluTanhActivationArch get_current_gelu_tanh_activation_arch()
  {
    static const GeluTanhActivationArch arch = []
    {
      int device_id = 0;
      cudaDeviceProp device_prop;
      cudaGetDevice(&device_id);
      cudaGetDeviceProperties(&device_prop, device_id);

      if (std::strncmp(device_prop.gcnArchName, "gfx936", 6) == 0)
      {
        return GeluTanhActivationArch::Gfx936;
      }
      if (std::strncmp(device_prop.gcnArchName, "gfx938", 6) == 0)
      {
        return GeluTanhActivationArch::Gfx938;
      }
      return GeluTanhActivationArch::Other;
    }();
    return arch;
  }

  inline bool pick_gfx938_gelu_tanh_activation_rows_per_block(
      GeluTanhActivationArch arch, int M, int block_n, int N, int &rows_per_block)
  {
    if (arch != GeluTanhActivationArch::Gfx938)
    {
      return false;
    }
    if (count_n_tiles(N, block_n) == 1 && N >= 1024)
    {
      if (M >= 4096)
      {
        rows_per_block = 4;
        return true;
      }
      if (M >= 1024)
      {
        rows_per_block = 8;
        return true;
      }
      if (M >= 512)
      {
        rows_per_block = 4;
        return true;
      }
    }
    return false;
  }

  inline bool pick_gfx936_gelu_tanh_activation_rows_per_block(
      GeluTanhActivationArch arch, int M, int block_n, int N, int &rows_per_block)
  {
    if (arch != GeluTanhActivationArch::Gfx936)
    {
      return false;
    }
    if (count_n_tiles(N, block_n) == 1 && N >= 1024)
    {
      if (M == 512 || M == 1024)
      {
        rows_per_block = 8;
        return true;
      }
      if (M >= 768 && M <= 2048)
      {
        rows_per_block = 4;
        return true;
      }
    }
    return false;
  }

  template <int ACTIVATION>
  inline int pick_activation_rows_per_block(
      int M, int rows_per_block, int block_n, int N)
  {
    if (rows_per_block != 1)
    {
      return rows_per_block;
    }
    if (M <= 16)
    {
      if (M <= 2)
      {
        return 8;
      }
      if (M <= 4)
      {
        return 4;
      }
      return 2;
    }
    if constexpr (ACTIVATION == 4)
    {
      const GeluTanhActivationArch arch = get_current_gelu_tanh_activation_arch();
      int gfx936_rows_per_block = 1;
      if (pick_gfx936_gelu_tanh_activation_rows_per_block(
              arch, M, block_n, N, gfx936_rows_per_block))
      {
        return gfx936_rows_per_block;
      }
      int gfx938_rows_per_block = 1;
      if (pick_gfx938_gelu_tanh_activation_rows_per_block(
              arch, M, block_n, N, gfx938_rows_per_block))
      {
        return gfx938_rows_per_block;
      }
    }
    if (M >= 4096 && count_n_tiles(N, block_n) == 1)
    {
      if constexpr (ACTIVATION == 4)
      {
        if (N <= 512)
        {
          if (M >= 16384)
          {
            return 4;
          }
          if (M >= 8192)
          {
            return 8;
          }
          return 2;
        }
        if (M >= 16384)
        {
          return 4;
        }
        return 2;
      }
      else
      {
        if (N <= 512 && M < 16384)
        {
          return 2;
        }
        return 1;
      }
    }
    if (M >= 64)
    {
      return 2;
    }
    return 1;
  }

  template <int ACTIVATION>
  inline SiluLaunchConfig pick_activation_launch_config(
      int M, int N, int rows_per_block, int vec_size)
  {
    if (M >= 4096 && N <= 512 && vec_size != 1 && is_exact_n_tile_fit(N, 4))
    {
      if constexpr (ACTIVATION == 3)
      {
        if (M >= 16384)
        {
          SiluLaunchConfig cfg =
              pick_glu_launch_config<ACTIVATION>(M, N, rows_per_block, vec_size);
          cfg.rows_per_block =
              pick_activation_rows_per_block<ACTIVATION>(M, rows_per_block, cfg.block_n, N);
          return cfg;
        }
      }
      SiluLaunchConfig cfg;
      cfg.block_n = N;
      cfg.vec_size = 4;
      cfg.rows_per_block =
          pick_activation_rows_per_block<ACTIVATION>(M, rows_per_block, cfg.block_n, N);
      return cfg;
    }

    SiluLaunchConfig cfg =
        pick_glu_launch_config<ACTIVATION>(M, N, rows_per_block, vec_size);
    cfg.rows_per_block =
        pick_activation_rows_per_block<ACTIVATION>(M, rows_per_block, cfg.block_n, N);
    return cfg;
  }

  inline void check_glu_args(
      torch::Tensor &out, torch::Tensor &input, const char *op_name,
      int &M, int &N)
  {
    TORCH_CHECK(input.is_cuda(), op_name, ": input must be CUDA tensor");
    TORCH_CHECK(out.is_cuda(), op_name, ": out must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), op_name, ": input must be contiguous");
    TORCH_CHECK(out.is_contiguous(), op_name, ": out must be contiguous");
    TORCH_CHECK(input.scalar_type() == out.scalar_type(),
                op_name, ": input and out dtype mismatch");
    TORCH_CHECK(input.dim() >= 1, op_name, ": input dim must be >= 1");
    TORCH_CHECK(input.size(-1) % 2 == 0,
                op_name, ": input last dim must be even");
    TORCH_CHECK(out.dim() == input.dim(), op_name, ": rank mismatch");
    TORCH_CHECK(out.size(-1) * 2 == input.size(-1),
                op_name, ": out last dim should be input last dim / 2");
    for (int64_t i = 0; i < input.dim() - 1; ++i)
    {
      TORCH_CHECK(out.size(i) == input.size(i),
                  op_name, ": shape mismatch at dim ", i);
    }

    const int64_t M64 = input.numel() / input.size(-1);
    const int64_t N64 = input.size(-1) / 2;
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX,
                op_name, ": shape too large");
    M = static_cast<int>(M64);
    N = static_cast<int>(N64);
  }

  inline void check_activation_args(
      torch::Tensor &out, torch::Tensor &input, const char *op_name,
      int &M, int &N)
  {
    TORCH_CHECK(input.is_cuda(), op_name, ": input must be CUDA tensor");
    TORCH_CHECK(out.is_cuda(), op_name, ": out must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), op_name, ": input must be contiguous");
    TORCH_CHECK(out.is_contiguous(), op_name, ": out must be contiguous");
    TORCH_CHECK(input.scalar_type() == out.scalar_type(),
                op_name, ": input and out dtype mismatch");
    TORCH_CHECK(input.dim() >= 1, op_name, ": input dim must be >= 1");
    TORCH_CHECK(out.dim() == input.dim(), op_name, ": rank mismatch");
    for (int64_t i = 0; i < input.dim(); ++i)
    {
      TORCH_CHECK(out.size(i) == input.size(i),
                  op_name, ": shape mismatch at dim ", i);
    }

    const int64_t N64 = input.size(-1);
    const int64_t M64 = input.numel() / input.size(-1);
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX,
                op_name, ": shape too large");
    M = static_cast<int>(M64);
    N = static_cast<int>(N64);
  }

  inline void check_glu_launch_args(
      int rows_per_block, int vec_size, const char *op_name)
  {
    TORCH_CHECK(vec_size == 1 || vec_size == 2 || vec_size == 4,
                op_name, ": vec_size must be 1, 2, or 4");

    TORCH_CHECK(rows_per_block == 1 || rows_per_block == 2 ||
                    rows_per_block == 4 || rows_per_block == 8,
                op_name, ": rows_per_block must be 1, 2, 4, or 8");
  }

  template <int ACTIVATION>
  inline void glu_impl(
      torch::Tensor &out, torch::Tensor &input, int rows_per_block, int vec_size,
      float beta1, float beta2, const char *op_name)
  {
    int M = 0;
    int N = 0;
    check_glu_args(out, input, op_name, M, N);
    if (M == 0 || N == 0)
    {
      return;
    }
    check_glu_launch_args(rows_per_block, vec_size, op_name);
    const SiluLaunchConfig cfg =
        pick_glu_launch_config<ACTIVATION>(M, N, rows_per_block, vec_size);
    const int block_n = cfg.block_n;
    const int rows = cfg.rows_per_block;
    const int vec = cfg.vec_size;

    TORCH_CHECK(block_n % vec == 0,
                op_name, ": block_n=", block_n, " must be divisible by vec_size=",
                vec);
    TORCH_CHECK((block_n / vec) % kWarpSize == 0,
                op_name, ": threads per block must be a multiple of warpSize");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr const char *dispatch_name =
        (ACTIVATION == 0) ? "silu_and_mul" : ((ACTIVATION == 3) ? "gelu_and_mul" : ((ACTIVATION == 4) ? "gelu_tanh_and_mul" : "situ_glu"));
    MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), dispatch_name, [&]
                                {
    auto* out_ptr = out.data_ptr<scalar_t>();
    const auto* in_ptr = input.data_ptr<scalar_t>();

    switch (block_n) {
      case 128:
        launch_glu_rows<scalar_t, 128, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      case 256:
        launch_glu_rows<scalar_t, 256, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      case 384:
        launch_glu_rows<scalar_t, 384, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      case 512:
        launch_glu_rows<scalar_t, 512, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      case 768:
        launch_glu_rows<scalar_t, 768, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      case 1024:
        launch_glu_rows<scalar_t, 1024, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      case 1536:
        launch_glu_rows<scalar_t, 1536, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      case 2048:
        launch_glu_rows<scalar_t, 2048, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows, beta1, beta2);
        break;
      default:
        TORCH_CHECK(false, op_name, ": unsupported block_n=", block_n);
    } });
  }

  template <int ACTIVATION>
  inline void activation_impl(
      torch::Tensor &out, torch::Tensor &input, int rows_per_block, int vec_size,
      const char *op_name)
  {
    int M = 0;
    int N = 0;
    check_activation_args(out, input, op_name, M, N);
    if (M == 0 || N == 0)
    {
      return;
    }
    check_glu_launch_args(rows_per_block, vec_size, op_name);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    const SiluLaunchConfig cfg =
        pick_activation_launch_config<ACTIVATION>(M, N, rows_per_block, vec_size);
    const int block_n = cfg.block_n;
    const int rows = cfg.rows_per_block;
    const int vec = cfg.vec_size;

    TORCH_CHECK(block_n % vec == 0,
                op_name, ": block_n=", block_n, " must be divisible by vec_size=",
                vec);
    TORCH_CHECK((block_n / vec) % kWarpSize == 0,
                op_name, ": threads per block must be a multiple of warpSize");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr const char *dispatch_name =
        (ACTIVATION == 3) ? "gelu" : "gelu_tanh";
    MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), dispatch_name, [&]
                                {
    auto* out_ptr = out.data_ptr<scalar_t>();
    const auto* in_ptr = input.data_ptr<scalar_t>();

    switch (block_n) {
      case 128:
        launch_activation_rows<scalar_t, 128, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      case 256:
        launch_activation_rows<scalar_t, 256, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      case 384:
        launch_activation_rows<scalar_t, 384, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      case 512:
        launch_activation_rows<scalar_t, 512, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      case 768:
        launch_activation_rows<scalar_t, 768, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      case 1024:
        launch_activation_rows<scalar_t, 1024, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      case 1536:
        launch_activation_rows<scalar_t, 1536, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      case 2048:
        launch_activation_rows<scalar_t, 2048, ACTIVATION>(
            out_ptr, in_ptr, M, N, stream, vec, rows);
        break;
      default:
        TORCH_CHECK(false, op_name, ": unsupported block_n=", block_n);
    } });
  }

  inline void silu_and_mul(torch::Tensor &out, torch::Tensor &input,
                           int rows_per_block = 1, int vec_size = 2)
  {
    glu_impl<0>(
        out, input, rows_per_block, vec_size, 0.0f, 0.0f, "silu_and_mul");
  }

  inline void gelu_and_mul(torch::Tensor &out, torch::Tensor &input,
                           int rows_per_block = 1, int vec_size = 2)
  {
    glu_impl<3>(
        out, input, rows_per_block, vec_size, 0.0f, 0.0f, "gelu_and_mul");
  }

  inline void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input,
                                int rows_per_block = 1, int vec_size = 2)
  {
    glu_impl<4>(
        out, input, rows_per_block, vec_size, 0.0f, 0.0f, "gelu_tanh_and_mul");
  }

  inline void gelu(torch::Tensor &out, torch::Tensor &input,
                   int rows_per_block = 1, int vec_size = 2)
  {
    activation_impl<3>(
        out, input, rows_per_block, vec_size, "gelu");
  }

  inline void gelu_tanh(torch::Tensor &out, torch::Tensor &input,
                        int rows_per_block = 1, int vec_size = 2)
  {
    activation_impl<4>(
        out, input, rows_per_block, vec_size, "gelu_tanh");
  }

  inline void situ_glu(torch::Tensor &out, torch::Tensor &input,
                       float beta1 = 4.0f, float beta2 = 25.0f,
                       int rows_per_block = 1, int vec_size = 2)
  {
    TORCH_CHECK(beta1 > 0.0f, "situ_glu: beta1 must be positive");
    TORCH_CHECK(beta2 > 0.0f, "situ_glu: beta2 must be positive");
    if (beta1 == 4.0f && beta2 == 25.0f)
    {
      glu_impl<2>(
          out, input, rows_per_block, vec_size, beta1, beta2, "situ_glu");
    }
    else
    {
      glu_impl<1>(
          out, input, rows_per_block, vec_size, beta1, beta2, "situ_glu");
    }
  }

} // namespace moe_c
