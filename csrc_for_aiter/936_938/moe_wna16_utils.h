// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#pragma once
#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"
#include <Python.h>
using bhalf_t = __hip_bfloat16;
using half_t = __half;
using BFloat16 = bhalf_t;
using Float16 = half_t;
using Int32 = int;
using Int16 = unsigned short;
using Float32 = float;
using f8_t = uint8_t;

using vec4_fp16 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using vec8_fp16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using vec2_fp16 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using vec16_fp16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;

using vec4_bf16 = __attribute__((__vector_size__(4 * sizeof(unsigned short)))) unsigned short;
using vec8_bf16 = __attribute__((__vector_size__(8 * sizeof(unsigned short)))) unsigned short;
using vec2_bf16 = __attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short;
using vec16_bf16 = __attribute__((__vector_size__(16 * sizeof(unsigned short)))) unsigned short;

using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using vec2_uint = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;

using vec4_fp8 = __attribute__((__vector_size__(4 * sizeof(uint8_t)))) uint8_t;
using vec8_fp8 = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;
using vec2_fp8 = __attribute__((__vector_size__(2 * sizeof(uint8_t)))) uint8_t;
using vec16_fp8 = __attribute__((__vector_size__(16 * sizeof(uint8_t)))) uint8_t;

using vec4_int8 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;
using vec8_int8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using vec2_int8 = __attribute__((__vector_size__(2 * sizeof(int8_t)))) int8_t;
using vec16_int8 = __attribute__((__vector_size__(16 * sizeof(int8_t)))) int8_t;

using vec4_int32 = __attribute__((__vector_size__(4 * sizeof(int)))) int;

using __builtin_half2 = __attribute__((ext_vector_type(2))) __fp16;
using __float2 = __attribute__((ext_vector_type(2))) float;

using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;

using __bf16x2 = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;

template <typename scalar_t, size_t len>
using vec = __attribute__((__vector_size__(len * sizeof(scalar_t)))) scalar_t;

template <typename scalar_t>
union vec_element_8
{
};

template <>
union vec_element_8<bhalf_t>
{
  vec8_bf16 data;
};

template <>
union vec_element_8<__half>
{
  vec8_fp16 data;
};

template <typename scalar_t>
union vec_element_2
{
};

template <>
union vec_element_2<bhalf_t>
{
  vec2_bf16 data;
};

template <>
union vec_element_2<__half>
{
  vec2_fp16 data;
};

template <typename scalar_t, size_t len>
union union_vec
{
  uint8_t uint8_array[len * sizeof(scalar_t)];
  scalar_t scalar_array[len];
  float float_array[len * sizeof(scalar_t) / 4];
  uint32_t uint32_array[len * sizeof(scalar_t) / 4];
  vec<float, 2> float2_array[len * sizeof(scalar_t) / 8];
  vec<float, 4> float4_array[len * sizeof(scalar_t) / 16];
  half_t half_array[len * sizeof(scalar_t) / 2];
};

template <class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_load_dwordx2(DataType &v_data, int v_offset, vec<uint, 4> global_addr, int s_offset)
{

  int v_offset_bytes = v_offset << shfl_count;
  int s_offset_bytes = s_offset << shfl_count;

  asm volatile(
      "buffer_load_dwordx2 %0, %1, %2, %3 offen\n"
      : "=v"(v_data)
      : "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bytes)
      : "memory");
}

template <class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_load_dwordx4(DataType &v_data, int v_offset, vec<uint, 4> global_addr, int s_offset)
{

  int v_offset_bytes = v_offset << shfl_count;
  int s_offset_bytes = s_offset << shfl_count;

  asm volatile(
      "buffer_load_dwordx4 %0, %1, %2, %3 offen\n"
      : "=v"(v_data)
      : "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bytes)
      : "memory");
}

template <typename scalar_t>
class ScalarType
{
};

template <>
class ScalarType<half>
{
public:
  using scalar_t = half;
  using scalar_t2 = half2;

  static __device__ float inline num2float(const half x)
  {
    return __half2float(x);
  }

  static __device__ half2 inline num2num2(const half x)
  {
    return __half2half2(x);
  }

  static __device__ half2 inline nums2num2(const half x1, const half x2)
  {
    return __halves2half2(x1, x2);
  }

  static __host__ __device__ half inline float2num(const float x)
  {
    return __float2half(x);
  }

  static __host__ __device__ half inline int2num(const float x)
  {
    return __int2half_rn(x);
  }

  static __host__ __device__ float2 inline num22float2(const half2 x)
  {
    return __half22float2(x);
  }

  static __host__ __device__ half2 inline float22num2(const float2 x)
  {
    return __float22half2_rn(x);
  }
};

template <>
class ScalarType<__hip_bfloat16>
{
public:
  using scalar_t = __hip_bfloat16;
  using scalar_t2 = __hip_bfloat162;

  static __device__ float inline num2float(const __hip_bfloat16 x)
  {
    return __bfloat162float(x);
  }

  static __device__ __hip_bfloat162 inline num2num2(const __hip_bfloat16 x)
  {
    return __bfloat162bfloat162(x);
  }

  static __device__ __hip_bfloat162 inline nums2num2(const __hip_bfloat16 x1,
                                                     const __hip_bfloat16 x2)
  {
    return __halves2bfloat162(x1, x2);
  }

  static __host__ __device__ __hip_bfloat16 inline float2num(const float x)
  {
    return __float2bfloat16(x);
  }

  static __host__ __device__ __hip_bfloat16 inline int2num(const int x)
  {
    return __float2bfloat16(static_cast<float>(x));
  }

  static __host__ __device__ float2 inline num22float2(const __hip_bfloat162 x)
  {
    return __bfloat1622float2(x);
  }

  static __host__ __device__ __hip_bfloat162 inline float22num2(const float2 x)
  {
    return __float22bfloat162_rn(x);
  }
};

template <int lut>
__device__ inline int lop3(int a, int b, int c)
{

  return a & b | c;
}

template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a)
{
  constexpr uint32_t mask0 = (mask >> 0) & 0x0F;
  constexpr uint32_t mask1 = (mask >> 4) & 0x0F;
  constexpr uint32_t mask2 = (mask >> 8) & 0x0F;
  constexpr uint32_t mask3 = (mask >> 12) & 0x0F;

  constexpr uint32_t s0 = (mask0 < 4) ? 0 : (start_byte >> ((mask0 - 4) * 8));
  constexpr uint32_t s1 = (mask1 < 4) ? 0 : (start_byte >> ((mask1 - 4) * 8));
  constexpr uint32_t s2 = (mask2 < 4) ? 0 : (start_byte >> ((mask2 - 4) * 8));
  constexpr uint32_t s3 = (mask3 < 4) ? 0 : (start_byte >> ((mask3 - 4) * 8));

  uint32_t b0 = (mask0 < 4) ? (a >> (mask0 * 8)) : s0;
  uint32_t b1 = (mask1 < 4) ? (a >> (mask1 * 8)) : s1;
  uint32_t b2 = (mask2 < 4) ? (a >> (mask2 * 8)) : s2;
  uint32_t b3 = (mask3 < 4) ? (a >> (mask3 * 8)) : s3;

  return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

template <typename scalar_t2, int bit>
__device__ inline void dequant(int q, scalar_t2 *res) {}

template <>
__device__ inline void dequant<half2, 4>(int q, half2 *res)
{
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  q >>= 8;
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  res[0] = __hsub2(*reinterpret_cast<half2 *>(&lo0),
                   *reinterpret_cast<const half2 *>(&SUB));
  res[1] = __hfma2(*reinterpret_cast<half2 *>(&hi0),
                   *reinterpret_cast<const half2 *>(&MUL),
                   *reinterpret_cast<const half2 *>(&ADD));
  res[2] = __hsub2(*reinterpret_cast<half2 *>(&lo1),
                   *reinterpret_cast<const half2 *>(&SUB));
  res[3] = __hfma2(*reinterpret_cast<half2 *>(&hi1),
                   *reinterpret_cast<const half2 *>(&MUL),
                   *reinterpret_cast<const half2 *>(&ADD));
}

template <>
__device__ inline void dequant<half2, 8>(int q, half2 *res)
{
  static constexpr uint32_t mask_for_elt_01 = 0x5250;
  static constexpr uint32_t mask_for_elt_23 = 0x5351;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;

  uint32_t lo = prmt<start_byte_for_fp16, mask_for_elt_01>(q);
  uint32_t hi = prmt<start_byte_for_fp16, mask_for_elt_23>(q);

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;

  res[0] = __hsub2(*reinterpret_cast<half2 *>(&lo),
                   *reinterpret_cast<const half2 *>(&I8s_TO_F16s_MAGIC_NUM));
  res[1] = __hsub2(*reinterpret_cast<half2 *>(&hi),
                   *reinterpret_cast<const half2 *>(&I8s_TO_F16s_MAGIC_NUM));
}

template <typename scalar_t2, int bit>
__device__ inline void dequant_block(int q, scalar_t2 *res) {}

template <>
__device__ inline void dequant_block<half2, 8>(int q, half2 *res)
{
  int8_t q0 = (q >> 0) & 0xFF;  // 第0字节
  int8_t q1 = (q >> 8) & 0xFF;  // 第1字节
  int8_t q2 = (q >> 16) & 0xFF; // 第2字节
  int8_t q3 = (q >> 24) & 0xFF; // 第3字节

  int16_t q0_ext = q0, q1_ext = q1, q2_ext = q2, q3_ext = q3;

  half2 h0 = __floats2half2_rn(q0_ext * 1.0f, q1_ext * 1.0f);

  half2 h1 = __floats2half2_rn(q2_ext * 1.0f, q3_ext * 1.0f);

  res[0] = h0;
  res[1] = h1;
}

template <>
__device__ inline void dequant<__hip_bfloat162, 4>(int q, __hip_bfloat162 *res)
{
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC300C300;
  using __bf16x2 = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;
  __bf16x2 mul_ = *(__bf16x2 *)&MUL;
  __bf16x2 add_ = *(__bf16x2 *)&ADD;
  __bf16x2 lo0_ = *(__bf16x2 *)&lo0;
  __bf16x2 hi0_ = *(__bf16x2 *)&hi0;
  __bf16x2 lo1_ = *(__bf16x2 *)&lo1;
  __bf16x2 hi1_ = *(__bf16x2 *)&hi1;
  __bf16x2 tmp0 = (__bf16x2)((__bf16x2)(lo0_ * mul_) + add_);
  __bf16x2 tmp1 = (__bf16x2)((__bf16x2)(hi0_ * mul_) + add_);
  __bf16x2 tmp2 = (__bf16x2)((__bf16x2)(lo1_ * mul_) + add_);
  __bf16x2 tmp3 = (__bf16x2)((__bf16x2)(hi1_ * mul_) + add_);
  res[0] = *(__hip_bfloat162 *)&tmp0;
  res[1] = *(__hip_bfloat162 *)&tmp1;
  res[2] = *(__hip_bfloat162 *)&tmp2;
  res[3] = *(__hip_bfloat162 *)&tmp3;
}

template <>
__device__ inline void dequant<__hip_bfloat162, 8>(int q, __hip_bfloat162 *res)
{
  float fp32_intermediates[4];
  uint32_t *fp32_intermediates_casted =
      reinterpret_cast<uint32_t *>(fp32_intermediates);

  static constexpr uint32_t fp32_base = 0x4B000000;
  fp32_intermediates_casted[0] = __byte_perm(q, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(q, fp32_base, 0x7652);
  fp32_intermediates_casted[2] = __byte_perm(q, fp32_base, 0x7651);
  fp32_intermediates_casted[3] = __byte_perm(q, fp32_base, 0x7653);

  fp32_intermediates[0] -= 8388608.f;
  fp32_intermediates[1] -= 8388608.f;
  fp32_intermediates[2] -= 8388608.f;
  fp32_intermediates[3] -= 8388608.f;

  uint32_t *bf16_result_ptr = reinterpret_cast<uint32_t *>(res);
  bf16_result_ptr[0] = __byte_perm(fp32_intermediates_casted[0],
                                   fp32_intermediates_casted[1], 0x7632);
  bf16_result_ptr[1] = __byte_perm(fp32_intermediates_casted[2],
                                   fp32_intermediates_casted[3], 0x7632);
}

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)()                                       \
  {                                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }
