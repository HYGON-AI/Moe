#pragma once
#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"
#include "hip/hip_fp8.h"
#include <type_traits>
using fp8_e4m3 = __hip_fp8_e4m3;
using fp8_e4m3x4 = __hip_fp8x4_e4m3;
using fp8_e5m2x4 = __hip_fp8x4_e5m2;
using bhalf_t = __hip_bfloat16;
using half = __half;
using BFloat16 = bhalf_t;
using Float16 = half;
using Int32 = int;
using Int16 = unsigned short;
using Float32 = float;
using f8_t = uint8_t;

using fp8x8_t = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;
using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using uint8x8_t = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;
using half2_t = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using half8_t = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using v4bh = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using floatx2 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using floatx2_bf16 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) char;
using uintx4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using BFloat16x4 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) char;
using intx2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int8x16 = __attribute__((__vector_size__(16 * sizeof(int8_t)))) char;
using bhalf_tx4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) Int16;
#ifndef AITER_FLOAT8_E4M3_T_DEFINED
#define AITER_FLOAT8_E4M3_T_DEFINED
struct alignas(1) Float8_e4m3_t
{
    uint8_t data;
    __host__ __device__ Float8_e4m3_t() = default;
    __host__ __device__ Float8_e4m3_t(uint8_t value) : data(value) {}
};
#endif

template <typename Element, size_t len>
struct w16_vec
{
    using type = __attribute__((__vector_size__(len * sizeof(Element)))) Element;
};

template <size_t len>
struct w16_vec<__half, len>
{
    using type = __attribute__((__vector_size__(len * sizeof(uint16_t)))) _Float16;
};

template <size_t len>
struct w16_vec<BFloat16, len>
{
    using type = __attribute__((__vector_size__(len * sizeof(uint16_t)))) unsigned short;
};

template <size_t len>
struct w16_vec<int8_t, len>
{
    using type = __attribute__((__vector_size__(len * sizeof(int8_t)))) char;
};

template <size_t len>
struct w16_vec<int, len>
{
    using type = __attribute__((__vector_size__(len * sizeof(int)))) int;
};
union alignas(16) int4_vec
{
    intx4 vec;   // 128bit 向量，用于 SIMD 读取
    int data[4]; // 4 个 int，用于逐个访问
};
template <size_t len>
union alignas(32) int8_vec
{
    intx4 vec[len / 4]; // 128bit 向量，用于 SIMD 读取
    int data[len];      // 4 个 int，用于逐个访问
};

template <size_t len, typename Element>
union f16_vec
{
    floatx2 floatx2_array[len / 4];
    floatx4 floatx4_array[len / 8];
    BFloat16x4 vec1[len / 4];
    BFloat16 data[len];
    typename w16_vec<Element, 4>::type floatx2_f16_array[len / 4];
    typename w16_vec<Element, 8>::type floatx4_f16_array[len / 8];
};

inline __device__ intx2 amd_assembly_i4_to_inx2(int a)
{
    uint32_t i4x8 = static_cast<uint32_t>(a);
    uint32_t fp8x4_0;
    uint32_t fp8x4_1;
    float tmp_0, tmp_1, tmp_2;
#if defined(__gfx938__)
    asm volatile("v_cvt_off_f32_i4 %[v_tmp_0], %[v_src]\n"
                 "v_cvt_off_f32_i4 %[v_tmp_1], %[v_src], src0_sel:BYTE_2\n"
                 "v_cvt_pk_fp8_f32 %[v_dst_0], %[v_tmp_0], %[v_tmp_1], %[v_tmp_1]\n"
                 "v_cvt_off_f32_i4 %[v_tmp_0], %[v_src], src0_sel:BYTE_1\n"
                 "v_cvt_off_f32_i4 %[v_tmp_1], %[v_src], src0_sel:BYTE_3\n"
                 "v_cvt_pk_fp8_f32 %[v_dst_1], %[v_tmp_0], %[v_tmp_1], %[v_tmp_1]\n"
                 "v_lshrrev_b32 %[v_tmp_2], 4, %[v_src]\n"
                 "v_cvt_off_f32_i4 %[v_tmp_0], %[v_tmp_2]\n"
                 "v_cvt_off_f32_i4 %[v_tmp_1], %[v_tmp_2], src0_sel:BYTE_2\n"
                 "v_cvt_pk_fp8_f32 %[v_dst_0], %[v_tmp_0], %[v_tmp_1], %[v_tmp_1], op_sel:[0, 0, 1]\n"
                 "v_cvt_off_f32_i4 %[v_tmp_0], %[v_tmp_2], src0_sel:BYTE_1\n"
                 "v_cvt_off_f32_i4 %[v_tmp_1], %[v_tmp_2], src0_sel:BYTE_3\n"
                 "v_cvt_pk_fp8_f32 %[v_dst_1], %[v_tmp_0], %[v_tmp_1], %[v_tmp_1], op_sel:[0, 0, 1]\n"
                 : [v_tmp_0] "+v"(tmp_0),
                   [v_tmp_1] "+v"(tmp_1),
                   [v_tmp_2] "+v"(tmp_2),
                   [v_dst_0] "+v"(fp8x4_0),
                   [v_dst_1] "+v"(fp8x4_1),
                   [v_src] "+v"(i4x8)
                 :);
#endif
    return (intx2){static_cast<int>(fp8x4_0), static_cast<int>(fp8x4_1)};
}

template <typename Element, size_t len>
union union_vec_fp8
{
    uint8_t uint8_array[len * sizeof(Element)];
    int8_t int8_array[len * sizeof(Element)];
    uint8x8_t uint8x8_array[len * sizeof(Element) / 8];
    Element scalar_array[len];
    int int_array[len * sizeof(Element) / 4];
    float float_array[len * sizeof(Element) / 4];
    int32_t uint_array[len * sizeof(Element) / 4];
    int64_t uint64_array[len * sizeof(Element) / 8];
    intx2 int2_array[len * sizeof(Element) / 8];
    intx4 int4_array[len * sizeof(Element) / 16];
    floatx4 float4_array[len * sizeof(Element) / 16];
};

template <typename Element, size_t len>
union w16_union_vec
{
    int8_t int8_array[len * sizeof(Element)];
    int8x8_t int8x8_array[len * sizeof(Element) / 8];
    Element scalar_array[len];
    typename w16_vec<Element, 2>::type scalar2_array[len / 2];
    int int_array[len * sizeof(Element) / 4];
    float float_array[len * sizeof(Element) / 4];
    int32_t uint_array[len * sizeof(Element) / 4];
    int64_t uint64_array[len * sizeof(Element) / 8];
    w16_vec<int8_t, 8>::type int8t_array[len * sizeof(Element) / 8];
    w16_vec<int, 2>::type int2_array[len * sizeof(Element) / 8];
    w16_vec<int, 4>::type int4_array[len * sizeof(Element) / 16];
    w16_vec<float, 4>::type float4_array[len * sizeof(Element) / 16];
};

template <const int kHeadDim, typename T>
__device__ __forceinline__ typename w16_vec<uint, 4>::type w16_tcp_cache_swizzle_func(const T *ptr)
{
    typename w16_vec<uint, 4>::type res;
    *(uint64_t *)&res = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
    if constexpr (kHeadDim == 196)
    {
        res[1] += 0x41800000; // 62 bit: cache swizzle;  48~61: Stride
    }
    else if constexpr (kHeadDim == 128)
    {
        res[1] += 0x41000000; // stride 256 Bytes and change tagram
    }
    else if constexpr (kHeadDim == 64)
    {                         // 会走这里
        res[1] += 0x40800000; // stride 128 Bytes and change tagram
    }
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    return res;
}
#define S_BARRIER                      \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile("s_barrier");         \
    __builtin_amdgcn_sched_barrier(0);
#define vmcnt_wait(X)                  \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(                      \
        "s_waitcnt vmcnt(%0)\n\t"      \
        "s_barrier\n" ::"I"(X)         \
        :);                            \
    __builtin_amdgcn_sched_barrier(0);

#define vmcnt(X)                           \
    __builtin_amdgcn_sched_barrier(0);     \
    asm volatile(                          \
        "s_waitcnt vmcnt(%0)\n\t" ::"I"(X) \
        :);                                \
    __builtin_amdgcn_sched_barrier(0);

#define lgkmcnt_wait(X)                               \
    __builtin_amdgcn_sched_barrier(0);                \
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "I"(X)); \
    __builtin_amdgcn_sched_barrier(0);
template <typename VEC, typename dwordx2>
__forceinline__ __device__ void w16_inline_ds_read2_b32_no_wait(VEC *shared_addr, const int &lds_offset, dwordx2 &reg_val, const int offset1)
{
    int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
    asm volatile(
        "ds_read2_b32 %0 ,%1 offset0:0 offset1:%2\n"
        : "=v"(reg_val)
        : "v"(ldsAddr), "B"(offset1)
        :);
}

template <typename VEC, typename dword>
__forceinline__ __device__ void inline_ds_read_b32_no_wait(VEC *shared_addr, const int &lds_offset, dword &reg_val, const int offset1)
{
    int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
    asm volatile(
        "ds_read_b32 %0 ,%1 offset0:0 offset1:%2\n"
        : "=v"(reg_val)
        : "v"(ldsAddr), "B"(offset1)
        :);
}

template <typename VEC, typename dwordx4>
__forceinline__ __device__ void inline_ds_read_b128_no_wait(VEC *shared_addr, dwordx4 &reg_val)
{
    int ldsAddr = reinterpret_cast<size_t>(shared_addr);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "\n ds_read_b128 %0 ,%1\n"
        : "=v"(reg_val)
        : "v"(ldsAddr)
        :);
    __builtin_amdgcn_sched_barrier(0);
}

template <typename VEC, typename dwordx4>
__forceinline__ __device__ void inline_ds_read_b128_no_wait_marlin(VEC *shared_addr, const int &lds_offset, dwordx4 &reg_val)
{
    int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset;
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "ds_read_b128 %0, %1\n"
        : "=v"(reg_val)
        : "v"(ldsAddr)
        :);
    __builtin_amdgcn_sched_barrier(0);
}

#define lgkmcnt_wait_barrier(X)        \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(                      \
        "s_waitcnt lgkmcnt(%0)\n\t"    \
        "s_barrier\n" ::"I"(X)         \
        :);                            \
    __builtin_amdgcn_sched_barrier(0);

__device__ __forceinline__ int w16_cvta_to_shared(const void *ptr)
{
    return (reinterpret_cast<size_t>(ptr) & 0xFFFFFFFF);
}
__device__ __forceinline__ void w16_cp_async4(float *smem_ptr, uintx4 global_addr, const int gvOffset_s, const int &gvOffset_v)
{
    int smem_offset = w16_cvta_to_shared(smem_ptr); // reinterpret_cast<size_t>(smem_ptr);
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(gvOffset_v),
        "s"(smem_offset), "s"(global_addr), "s"(gvOffset_s)
        :);
}
template <typename src_type = bhalf_t, typename dst_type = bhalf_t, const int dword_count = 2, const int auxilariy = 0>
__forceinline__ __device__ void builtin_buffer_load_dword_lds2(src_type *const shared_addr, const typename w16_vec<uint, 4>::type rsrc, const int &lds_offset, const int gvOffset_s, const int &gvOffset_v)
{
    constexpr int bytes_per_element = sizeof(float);
    dst_type *ptr = reinterpret_cast<dst_type *>(shared_addr) + lds_offset * 2;
    __builtin_amdgcn_raw_buffer_load_lds(
        rsrc,
        (__attribute__((address_space(3))) int *)ptr,
        dword_count * 4, // dword读取
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0,        /* immediate offset, instruction offset */
        auxilariy /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
}
template <typename src_type = int8_t, typename dst_type = float, const int dword_count = 4, const int auxilariy = 0>
__forceinline__ __device__ void builtin_buffer_load_dword_lds4(src_type *const shared_addr, const typename w16_vec<uint, 4>::type rsrc, const int &lds_offset, const int gvOffset_s, const int &gvOffset_v)
{
    constexpr int bytes_per_element = sizeof(dst_type);
    dst_type *ptr = reinterpret_cast<dst_type *>(shared_addr) + lds_offset;
    __builtin_amdgcn_raw_buffer_load_lds(
        rsrc,
        (__attribute__((address_space(3))) int *)ptr,
        dword_count * 4, // dword读取
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0,        /* immediate offset, instruction offset */
        auxilariy /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
}
template <typename src_type = bhalf_t, typename dst_type = float, const int dword_count = 1, const int auxilariy = 0>
__forceinline__ __device__ void w16_builtin_buffer_load_dword_lds(src_type *const shared_addr, const typename w16_vec<uint, 4>::type rsrc, const int &lds_offset, const int gvOffset_s, const int &gvOffset_v)
{
    constexpr int bytes_per_element = sizeof(dst_type);
    dst_type *ptr = reinterpret_cast<dst_type *>(shared_addr) + lds_offset;
    __builtin_amdgcn_raw_buffer_load_lds(
        rsrc,
        (__attribute__((address_space(3))) int *)ptr,
        dword_count * 4, // dword读取
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0,        /* immediate offset, instruction offset */
        auxilariy /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
}

template <typename T>
__forceinline__ __device__ void buffer_load_lds_dwordx4(const T *ptr, T *lds_ptr, int lds_offset, int offset_v, int offset_s)
{ // offset 代表global的offset
    intx4 global_ptr;
    *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
    global_ptr[1] += 0x40800000;                                // 0x40400000？ 0x40800000
    global_ptr[2] = 0x80000000;
    global_ptr[3] = 0x00020000;
    constexpr int bytes_per_element = sizeof(T);
    T *dst_ptr = reinterpret_cast<T *>(lds_ptr) + lds_offset;

    int smem_offset;
#if (defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__))
    smem_offset = __builtin_amdgcn_readfirstlane(w16_cvta_to_shared(dst_ptr));
#elif defined(__gfx938__)
    smem_offset = __builtin_hcu_readfirstlane(w16_cvta_to_shared(dst_ptr));
#endif
#if (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__))
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(smem_offset), "s"(global_ptr), "s"(offset_s)
        :);
    __builtin_amdgcn_sched_barrier(0);
#endif
}

template <typename T>
__forceinline__ __device__ void buffer_load_lds_dwordx1(const T *ptr, T *lds_ptr, int lds_offset, int offset_v, int offset_s)
{ // offset 代表global的offset
    intx4 global_ptr;
    *(uint64_t *)&global_ptr = (reinterpret_cast<uint64_t>(ptr)); // res[0]放首地址信息
    global_ptr[1] += 0x41000000;                                  // 0x40400000？ 0x40800000
    global_ptr[2] = 0x80000000;
    global_ptr[3] = 0x00020000;
    constexpr int bytes_per_element = sizeof(T);
    T *dst_ptr = reinterpret_cast<T *>(lds_ptr) + lds_offset;

    int smem_offset;
#if (defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__))
    smem_offset = __builtin_amdgcn_readfirstlane(w16_cvta_to_shared(dst_ptr));
#elif defined(__gfx938__)
    smem_offset = __builtin_hcu_readfirstlane(w16_cvta_to_shared(dst_ptr));
#endif
#if (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__))
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(smem_offset), "s"(global_ptr), "s"(offset_s)
        :);
    __builtin_amdgcn_sched_barrier(0);
#endif
}

template <typename T>
__forceinline__ __device__ void buffer_load_lds_dwordx2(const T *ptr, T *lds_ptr, int lds_offset, int offset_v, int offset_s)
{ // offset 代表global的offset
    intx4 global_ptr;
    *(uint64_t *)&global_ptr = (reinterpret_cast<uint64_t>(ptr)); // res[0]放首地址信息
    global_ptr[1] += 0x41000000;                                  // 0x40400000？ 0x40800000
    global_ptr[2] = 0x80000000;
    global_ptr[3] = 0x00020000;
    constexpr int bytes_per_element = sizeof(T);
    T *dst_ptr = reinterpret_cast<T *>(lds_ptr) + lds_offset;
    int smem_offset;
#if (defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__))
    smem_offset = __builtin_amdgcn_readfirstlane(w16_cvta_to_shared(dst_ptr));
#elif defined(__gfx938__)
    smem_offset = __builtin_hcu_readfirstlane(w16_cvta_to_shared(dst_ptr));
#endif

#if (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__))
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(smem_offset), "s"(global_ptr), "s"(offset_s)
        :);
    __builtin_amdgcn_sched_barrier(0);
#endif
}

template <typename T>
__device__ inline void amdgcn_buffer_load_dwordx4_lds(void *smem_ptr, const T *glob_ptr, int offset_v, const int offset_s)
{
    int smem_offset_sgpr;
#if (defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__))
    smem_offset_sgpr = __builtin_amdgcn_readfirstlane(w16_cvta_to_shared(smem_ptr));
#elif defined(__gfx938__)
    smem_offset_sgpr = __builtin_hcu_readfirstlane(w16_cvta_to_shared(smem_ptr));
#endif
    typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
    uint32x4_t global_addr = {0};
    *(uint64_t *)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
    global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000; // 0000 0000 0000 0010 0000 0000 0000 0000
#if (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__))
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(smem_offset_sgpr), "s"(global_addr), "s"(offset_s)
        :);
    __builtin_amdgcn_sched_barrier(0);
#endif
}

__device__ __forceinline__ void cp_async_buffer_load_dwordx4(void *smem_ptr, uintx4 global_addr, const int gvOffset_s, const int &gvOffset_v)
{
    int smem_offset_sgpr;
#if (defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__))
    smem_offset_sgpr = __builtin_amdgcn_readfirstlane(w16_cvta_to_shared(smem_ptr));
#elif defined(__gfx938__)
    smem_offset_sgpr = __builtin_hcu_readfirstlane(w16_cvta_to_shared(smem_ptr));
#endif
#if (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__))
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx4 %0, %2, %3, offen  offset:0, lds \n" ::"v"(gvOffset_v),
        "s"(smem_offset_sgpr), "s"(global_addr), "s"(gvOffset_s)
        :);
#endif
}

template <typename src_type = bhalf_t, typename dst_type = char, const int dword_count = 4, const int auxilariy = 0>
__forceinline__ __device__ void builtin_buffer_load_dwordx4_lds(src_type *const shared_addr, const typename w16_vec<uint, 4>::type rsrc, const int &lds_offset, const int gvOffset_s, const int &gvOffset_v)
{
    constexpr int bytes_per_element = sizeof(dst_type);
    dst_type *ptr = reinterpret_cast<dst_type *>(shared_addr) + lds_offset /*bytes*/;
    __builtin_amdgcn_raw_buffer_load_lds(
        rsrc,
        (__attribute__((address_space(3))) int *)ptr,
        dword_count * 4, // dword读取
        gvOffset_v,
        gvOffset_s,
        0,        /* immediate offset, instruction offset */
        auxilariy /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
}

#define s_setprio(X)                   \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(                      \
        "s_setprio %0\n\t" ::"I"(X)    \
        :);                            \
    __builtin_amdgcn_sched_barrier(0);

template <typename T>
__device__ inline void amdgcn_buffer_load_dwordx2_lds(void *smem_ptr, T *glob_ptr, int offset_v, int offset_s)
{
    int smem_offset_sgpr;
#if (defined(__gfx936__) || defined(__gfx928__) || defined(__gfx92a__))
    smem_offset_sgpr = __builtin_amdgcn_readfirstlane(w16_cvta_to_shared(smem_ptr));
#elif defined(__gfx938__)
    smem_offset_sgpr = __builtin_hcu_readfirstlane(w16_cvta_to_shared(smem_ptr));
#endif

    typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
    uint32x4_t global_addr = {0};
    *(uint64_t *)&global_addr = reinterpret_cast<uint64_t>(glob_ptr);
    global_addr[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000; // 0000 0000 0000 0010 0000 0000 0000 0000
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(smem_offset_sgpr), "s"(global_addr), "s"(offset_s)
        :);
    return;
}

template <typename T>
__forceinline__ __device__ intx4 builtin_amdgcn_buffer_load_reg_dwordx4(const T *ptr, const int vindex, const int offset)
{ // const int offset
    intx4 rsrc;
    *(uint64_t *)&rsrc = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
    rsrc[1] += 0x40800000;
    rsrc[2] = 0x80000000;
    rsrc[3] = 0x00020000;

    rsrc = __builtin_amdgcn_buffer_load_dwordx4(rsrc, vindex, offset, false, false);

    return rsrc;
}
template <typename T>
__forceinline__ __device__ intx2 builtin_amdgcn_buffer_load_reg_dwordx2(const T *ptr, const int vindex, const int offset)
{
    intx4 rsrc;
    intx2 rsrc1;
    *(uint64_t *)&rsrc = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
    rsrc[1] += 0x41000000;
    rsrc[2] = 0x80000000;
    rsrc[3] = 0x00020000;
    rsrc1 = __builtin_amdgcn_buffer_load_dwordx2(rsrc, vindex, offset, false, false);
    return rsrc1;
}

template <typename T>
__forceinline__ __device__ intx2 buffer_load_reg_dwordx2(const T *ptr1, int vindex)
{ // const int offset
    intx4 global_ptr;
    intx2 value;
    *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr1); // res[0]放首地址信息
    global_ptr[1] += 0x41000000;
    global_ptr[2] = 0x80000000;
    global_ptr[3] = 0x00020000;
    asm volatile("buffer_load_dwordx2 %0,%1,%2,0, offen offset:0 \n"
                 : "=v"(value), "+v"(vindex), "+s"(global_ptr));
    return value;
}
template <typename T>
__forceinline__ __device__ void w16_buffer_load_reg_dwordx4(const T *ptr, typename w16_vec<int, 4>::type &rsrc, const int vindex, int offset)
{ // const int offset

    intx4 global_ptr;
    *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
    global_ptr[1] += 0x40800000;
    global_ptr[2] = 0x80000000;
    global_ptr[3] = 0x00020000;
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("buffer_load_dwordx4 %0,%1,%2,0, offen offset:0 \n"
                 : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));
    __builtin_amdgcn_sched_barrier(0);
    return;
}

template <int CACHE_MODE, typename T>
__forceinline__ __device__ void w16_buffer_load_reg_dwordx4_cache(const T *ptr, typename w16_vec<int, 4>::type &rsrc, const int vindex, int offset)
{
    intx4 global_ptr;
    *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr);
    if constexpr (CACHE_MODE == 1)
    {
        global_ptr[1] += 0x40800000;
    }
    else if constexpr (CACHE_MODE == 2)
    {
        global_ptr[1] += 0x41000000;
    }
    else if constexpr (CACHE_MODE == 3)
    {
        global_ptr[1] += 0x42000000;
    }
    global_ptr[2] = 0x80000000;
    global_ptr[3] = 0x00020000;
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("buffer_load_dwordx4 %0,%1,%2,0, offen offset:0 \n"
                 : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));
    __builtin_amdgcn_sched_barrier(0);
    return;
}
inline __device__ constexpr int w16_ceil_div(int const &a, int const &b)
{
    return (a + b - 1) / b;
}

template <class Element>
__device__ floatx4 mmac(const typename w16_vec<Element, 4>::type &v1, const typename w16_vec<Element, 4>::type &v2, floatx4 &v3)
{
    floatx4 v4f;
#if (defined(__gfx936__) || defined(__gfx928__))
    v4f = __builtin_amdgcn_mmac_f32_16x16x16f16(v1, v2, v3);
#elif defined(__gfx938__)
    v4f = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(v1, v2, v3, false, false);
#elif defined(__gfx92a__)
    v4f = __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
    return v4f;
}

template <>
__device__ typename w16_vec<float, 4>::type mmac<half>(const typename w16_vec<half, 4>::type &v1, const typename w16_vec<half, 4>::type &v2, floatx4 &v3)
{
    floatx4 v4f;
#if (defined(__gfx936__) || defined(__gfx928__))
    v4f = __builtin_amdgcn_mmac_f32_16x16x16f16(v1, v2, v3);
#elif defined(__gfx938__)
    v4f = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(v1, v2, v3, false, false);
#elif defined(__gfx92a__)
    v4f = __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
    return v4f;
}

template <>
__device__ floatx4 mmac<bhalf_t>(const typename w16_vec<bhalf_t, 4>::type &v1, const typename w16_vec<bhalf_t, 4>::type &v2, floatx4 &v3)
{
    floatx4 v4f;
#if (defined(__gfx936__) || defined(__gfx928__))
    v4f = __builtin_amdgcn_mmac_f32_16x16x16bf16(v1, v2, v3);
#elif defined(__gfx938__)
    v4f = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(v1, v2, v3, false, false);
#elif defined(__gfx92a__)
    v4f = __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#endif
    return v4f;
}

#if !defined(__NVCC__)
constexpr int32_t w16_e5m2_exp_bits = 5;
constexpr int32_t w16_e5m2_mant_bits = 2;
constexpr int32_t w16_e5m2_bits = 8;
constexpr int32_t w16_e5m2_bias = (1 << (w16_e5m2_exp_bits - 1)) - 1;
constexpr int32_t w16_e4m3_exp_bits = 4;
constexpr int32_t w16_e4m3_mant_bits = 3;
constexpr int32_t w16_e4m3_bits = 8;
constexpr int32_t w16_e4m3_bias = (1 << (w16_e4m3_exp_bits - 1)) - 1;
constexpr int32_t w16_fp16_exp_bits = 5;
constexpr int32_t w16_fp16_mant_bits = 10;
constexpr int32_t w16_fp16_bits = 16;
constexpr int32_t w16_fp16_bias = (1 << (w16_fp16_exp_bits - 1)) - 1;
constexpr int32_t w16_fp32_exp_bits = 8;
constexpr int32_t w16_fp32_mant_bits = 23;
constexpr int32_t w16_fp32_bits = 32;
constexpr int32_t w16_fp32_bias = (1 << (w16_fp32_exp_bits - 1)) - 1;

__host__ __device__ static uint8_t w16_float2e4m3(const float src)
{
    uint32_t __src = *(unsigned int *)&src;
    uint8_t sign_bits = (__src & 0x80000000) >> (w16_fp32_bits - w16_e5m2_bits);
    uint32_t exp_bits = __src & 0x7f800000;
    uint32_t mant_bits = __src & 0x007fffff;
    uint32_t data_scale = __src & 0x7fffffff;
    /* NAN */
    uint8_t result = 0x0;
    if (exp_bits == 0x7f800000 and mant_bits > 0x0)
    {
        result = 0x7f; // for NV's __nv_cvt_float_to_fp8:cvt.rn.satfinite.e4m3x2.f32, output are all 0x7f
    }
    /* inf or greater than MAX value of E5M2 */
    else if ((exp_bits == 0x7f800000 and mant_bits == 0x0) or (data_scale > 0x43e00000))
    {
        result = sign_bits | 0x7e; // output MAX
    }
    /* less than MIN of denorm */
    else if (exp_bits <= 0x3a800000)
    {
        result = (exp_bits == 0x3a800000 and mant_bits > 0x0) ? sign_bits | 0x1 : sign_bits;
    }
    /* others */
    else
    {
        /* norm fp32 can be represented as denorm fp8 / norm */
        mant_bits = exp_bits < 0x3c800000 ? (0x800000 | mant_bits) >> ((0x3c800000 - exp_bits) >> w16_fp32_mant_bits) : mant_bits;
        exp_bits = exp_bits < 0x3c800000 ? 0x0 : ((exp_bits >> w16_fp32_mant_bits) - (w16_fp32_bias - w16_e4m3_bias)) << w16_e4m3_mant_bits;
        uint32_t discard_bits = mant_bits & 0xfffff;
        bool carry_a_bit = discard_bits > 0x80000 or (discard_bits == 0x80000 and (mant_bits & 0x100000));
        mant_bits = (mant_bits & 0x700000) >> (w16_fp32_mant_bits - w16_e4m3_mant_bits);
        mant_bits = carry_a_bit ? mant_bits + 1 : mant_bits;
        result = sign_bits + exp_bits + mant_bits; // + rather than |, since mant may carry a bit to exp
    }
    return result;
}

__host__ __device__ static float w16_e4m32float(const uint8_t src)
{
    float result;
    uint8_t __src = *(uint8_t *)&src;
    uint32_t sign_bits = __src & 0x80;
    uint32_t exp_bits = (__src & 0x78) >> w16_e4m3_mant_bits;
    uint32_t mant_bits = __src & 0x7;
    if (exp_bits == 0x0 and mant_bits >= 0x0)
    {
        result = 0.0078125f * ((mant_bits & 0x4) >> 2) + 0.00390625f * ((mant_bits & 0x2) >> 1) + 0.001953125f * (mant_bits & 0x1);
        result = sign_bits > 0 ? -result : result;
    }
    else
    {
        uint32_t tmp = (exp_bits == 0xf and mant_bits == 0x7)
                           ? /*input NaN*/ 0x7fffffff
                           : /*input norm*/ (sign_bits << (w16_fp32_bits - w16_e4m3_bits)) + ((exp_bits - w16_e4m3_bias + w16_fp32_bias) << w16_fp32_mant_bits) + (mant_bits << (w16_fp32_mant_bits - w16_e4m3_mant_bits));
        result = *(float *)&tmp;
    }
    return result;
}
#endif // end of #if !defined(__NVCC__)

static __device__ inline uint16_t w16_half_as_uint16(__half h)
{
    union
    {
        __half h;
        uint16_t u;
    } converter;
    converter.h = h;
    return converter.u;
}

static __device__ inline __half w16_uint16_as_half(uint16_t u)
{
    union
    {
        __half h;
        uint16_t u;
    } converter;
    converter.u = u;
    return converter.h;
}

template <typename T>
static __device__ inline uint16_t w16_f32_to_f16(float f)
{
    if constexpr (std::is_same<T, half>::value)
    {
        return w16_half_as_uint16(__float2half(f));
    }
    else
    {
        uint32_t u = *(uint32_t *)(&f);
        u += 0x7fff + ((u >> 16) & 1);
        return u >> 16;
    }
}

template <class FromType, class ToType, bool Is_short = false, typename std::enable_if<std::is_same<FromType, float>::value && std::is_same<ToType, half>::value, int>::type = 0>
__host__ __device__ ToType w16_DownCast(const FromType &from_var)
{
    return __float2half(from_var);
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType, float>::value && Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ uint8_t w16_DownCast(const float &from_var)
{
    return w16_float2e4m3(from_var);
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType, float>::value && !Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ Float8_e4m3_t w16_DownCast(const float &from_var)
{
    return Float8_e4m3_t(w16_float2e4m3(from_var));
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType, half>::value && Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ uint8_t w16_DownCast(const half &from_var)
{
    float src_f32 = __half2float(from_var);
    return w16_float2e4m3(src_f32);
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType, half>::value && !Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ Float8_e4m3_t w16_DownCast(const half &from_var)
{
    float src_f32 = __half2float(from_var);
    return Float8_e4m3_t(w16_float2e4m3(src_f32));
}

template <class FromType = half, class ToType = float, bool Is_short = false, typename std::enable_if<std::is_same<FromType, half>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float w16_UpCast(const half &from_var)
{
    return __half2float(from_var);
}
template <class FromType, class ToType, bool Is_short = false, typename std::enable_if<!Is_short && std::is_same<FromType, BFloat16>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float w16_UpCast(const BFloat16 &from_var)
{
    return __bfloat162float(from_var);
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float w16_UpCast(const uint8_t &from_var)
{
    return w16_e4m32float(from_var);
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<!Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float w16_UpCast(const Float8_e4m3_t &from_var)
{
    return w16_e4m32float(from_var.data);
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType, half>::value, int>::type = 0>
__host__ __device__ half w16_UpCast(const uint8_t &from_var)
{
    float src_f32 = w16_e4m32float(from_var);
    return __float2half(src_f32);
}
template <class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<!Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType, half>::value, int>::type = 0>
__host__ __device__ half w16_UpCast(const Float8_e4m3_t &from_var)
{
    float src_f32 = w16_e4m32float(from_var.data);
    return __float2half(src_f32);
}

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define DIV_ceil(x, size) (((x) - 1) / (size) + 1)
__device__ __forceinline__ void w16_GetBLockIdx(
    int32_t loop_idx, int32_t m_loop, int32_t n_loop, int32_t swizzl_direction, int32_t swizzl_count,
    int &m_idx, int &n_idx)
{
    int32_t in_batch_idx = loop_idx % (m_loop * n_loop);
    if (swizzl_direction == 0)
    { // Zn
        int32_t tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);
        int32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1)
        {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_row;
        n_idx = in_tile_block_idx / n_row;
    }
    else if (swizzl_direction == 1)
    { // Nz
        int32_t tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);
        int32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1)
        {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        n_idx = tile_block_idx * swizzl_count + (in_tile_block_idx % n_col);
        m_idx = (in_tile_block_idx / n_col);
    }
    else if (swizzl_direction == 2)
    { // Zz
        int32_t tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);
        int32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1)
        {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        int32_t n_tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
        int32_t n_tile_block_idx = in_tile_block_idx / (n_row * swizzl_count);
        int32_t n_in_tile_block_idx = in_tile_block_idx % (n_row * swizzl_count);
        int32_t n_col = swizzl_count;
        if (n_tile_block_idx == n_tile_block_loop - 1)
        {
            n_col = n_loop - swizzl_count * n_tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + n_in_tile_block_idx / n_col;
        n_idx = n_tile_block_idx * swizzl_count + (n_in_tile_block_idx % n_col);
    }
    else if (swizzl_direction == 3)
    { // Nn
        int32_t tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
        int32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        int32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);
        int32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1)
        {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        int32_t m_tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
        int32_t m_tile_block_idx = in_tile_block_idx / (n_col * swizzl_count);
        int32_t m_in_tile_block_idx = in_tile_block_idx % (n_col * swizzl_count);
        int32_t n_row = swizzl_count;
        if (m_tile_block_idx == m_tile_block_loop - 1)
        {
            n_row = m_loop - swizzl_count * m_tile_block_idx;
        }
        m_idx = m_tile_block_idx * swizzl_count + m_in_tile_block_idx % n_row;
        n_idx = tile_block_idx * swizzl_count + m_in_tile_block_idx / n_row;
    }
    else
    {
        int blocks_per_tile = n_loop * swizzl_count;
        int num_tiles = m_loop / swizzl_count;
        int block_idx_flatterned = n_idx * m_loop + m_idx;
        int tile_id = block_idx_flatterned / blocks_per_tile;
        int block_idx_in_tile = block_idx_flatterned % blocks_per_tile;
        int block_idx_x_in_tile = block_idx_in_tile % swizzl_count;
        int block_idx_y_in_tile = block_idx_in_tile / swizzl_count;
        if (m_idx >= num_tiles * swizzl_count)
        {
            int last_tile_dim_x = m_loop - num_tiles * swizzl_count;
            block_idx_x_in_tile = block_idx_in_tile % last_tile_dim_x;
            block_idx_y_in_tile = block_idx_in_tile / last_tile_dim_x;
        }

        int swizzled_block_idx_flatterned =
            block_idx_y_in_tile * m_loop + block_idx_x_in_tile + tile_id * swizzl_count;

        m_idx = swizzled_block_idx_flatterned % m_loop;
        n_idx = swizzled_block_idx_flatterned / m_loop;
    }
}
