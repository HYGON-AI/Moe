// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT


#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

#include "intrinsic_2.h"
#include "intrinsic.h"
#include "moe_wna16_utils.h"

namespace W4A16_UTILS
{
    namespace NORMAL
    {
        template <class T>
        inline __device__ vec4_fp32 mmac(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
        {
            return {0, 0, 0, 0};
        }

        template <>
        inline __device__ vec4_fp32 mmac<half_t>(const vec4_fp16 &v1, const vec4_fp16 &v2, const vec4_fp32 &v3)
        {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx928__)
            return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#else
            return {0, 0, 0, 0};
#endif
        }

        template <>
        inline __device__ vec4_fp32 mmac<__hip_bfloat16>(const vec4_bf16 &v1, const vec4_bf16 &v2, const vec4_fp32 &v3)
        {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx928__)
            return __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#else
            return {0, 0, 0, 0};
#endif
        }

    } // end namespace NORMAL

    namespace INTERLEAVE
    {
        template <class T>
        inline __device__ vec4_fp32 mmac_lit(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
        {
            return {0, 0, 0, 0};
        }

        template <>
        inline __device__ vec4_fp32 mmac_lit<__hip_bfloat16>(const vec4_bf16 &v1, const vec4_bf16 &v2, const vec4_fp32 &v3)
        {
            constexpr bool lit = true;
            constexpr bool lts = false;
#if defined(__gfx938__)
            return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(v1, v2, v3, lit, lts);
#else
            return {0, 0, 0, 0};
#endif
        }

        template <>
        inline __device__ vec4_fp32 mmac_lit<half_t>(const vec4_fp16 &v1, const vec4_fp16 &v2, const vec4_fp32 &v3)
        {
            constexpr bool lit = true;
            constexpr bool lts = false;
#if defined(__gfx938__)
            return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(v1, v2, v3, lit, lts);
#else
            return {0, 0, 0, 0};
#endif
        }

    } // end namespace INTERLEAVE

    template <typename T, bool INTERLEAVE>
    inline __device__ vec4_fp32 mmac(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
    {
        if constexpr (INTERLEAVE)
        {
            return INTERLEAVE::mmac_lit<T>(v1, v2, v3);
        }
        else
        {
            return NORMAL::mmac<T>(v1, v2, v3);
        }
    }

    namespace DATALOADER
    {
        template <typename T>
        __forceinline__ __device__ void buffer_load_reg_dwordx2_half4(const T *ptr, vec4_Element<T> &rsrc, const int vindex, int offset)
        { // const int offset

            intx4 global_ptr;
            *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
            global_ptr[1] += 0x50000000;
            global_ptr[2] = 0x80000000;
            global_ptr[3] = 0x00020000;

            asm volatile(
                "s_nop 2 \n\t"
                "buffer_load_dwordx2 %0,%1,%2,0, offen offset:0\n"
                : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));

            return;
        }

        template <typename T>
        __forceinline__ __device__ void buffer_load_reg_dword_half2(const T *ptr, vec2_Element<T> &rsrc, const int vindex, int offset)
        { // const int offset

            intx4 global_ptr;
            *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
            global_ptr[1] += 0x50000000;
            global_ptr[2] = 0x80000000;
            global_ptr[3] = 0x00020000;

            asm volatile(
                "s_nop 2 \n\t"
                "buffer_load_dword %0,%1,%2,0, offen offset:0\n"
                : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));

            return;
        }
        template <typename T>
        __forceinline__ __device__ void buffer_load_reg_dwordx4_half8(const T *ptr, vec8_Element<T> &rsrc, const int vindex, int offset)
        { // const int offset

            intx4 global_ptr;
            *(uint64_t *)&global_ptr = reinterpret_cast<uint64_t>(ptr); // res[0]放首地址信息
            global_ptr[1] += 0x50000000;
            global_ptr[2] = 0x80000000;
            global_ptr[3] = 0x00020000;

            asm volatile(
                "s_nop 2 \n\t"
                "buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n"
                : "=v"(rsrc), "+v"(offset), "+s"(global_ptr));

            return;
        }

    } // end namespace DATALOADER

    struct MMAC_IL
    {
        template <typename T>
        static inline __device__ vec4_fp32 matmul(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
        {
            return mmac_lit<T>(v1, v2, v3);
        }
    };

    struct MMAC_NORMAL
    {
        template <typename T>
        static inline __device__ vec4_fp32 matmul(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
        {
            return mmac<T>(v1, v2, v3);
        }
    };

    namespace ARCH
    {
        int getArch()
        {
            hipDeviceProp_t props;
            auto hipResult = hipGetDeviceProperties(&props, 0);
            std::string gcn_arch_name(props.gcnArchName);
            gcn_arch_name = gcn_arch_name.substr(3, 3);
            if (gcn_arch_name == "92a")
                gcn_arch_name = "930";
            int gcn_arch = std::stoi(gcn_arch_name);
            return gcn_arch;
        }

    } // end namespace ARCH

    namespace DEQUANTER
    {
        __device__ inline void dequant_weight_debug(int q, __bf16x2 scale, __hip_bfloat162 zero, __hip_bfloat162 *res)
        {

#if defined(__gfx938__)
            static constexpr uint32_t MASK = 0x000f000f;
            static constexpr uint32_t EX = 0x43004300;

            int lo0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
            q >>= 4;
            int hi0 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
            q >>= 4;
            int lo1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
            q >>= 4;
            int hi1 = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

            __bf16x2 scale_ = *(__bf16x2 *)&scale;
            __bf16x2 zero_ = *(__bf16x2 *)&zero;
            __bf16x2 lo0_ = *(__bf16x2 *)&lo0;
            __bf16x2 hi0_ = *(__bf16x2 *)&hi0;
            __bf16x2 lo1_ = *(__bf16x2 *)&lo1;
            __bf16x2 hi1_ = *(__bf16x2 *)&hi1;
            __bf16x2 tmp0 = (__bf16x2)((__bf16x2)(lo0_ + zero_) * scale_);
            __bf16x2 tmp1 = (__bf16x2)((__bf16x2)(hi0_ + zero_) * scale_);
            __bf16x2 tmp2 = (__bf16x2)((__bf16x2)(lo1_ + zero_) * scale_);
            __bf16x2 tmp3 = (__bf16x2)((__bf16x2)(hi1_ + zero_) * scale_);
            res[0] = *(__hip_bfloat162 *)&tmp0;
            res[1] = *(__hip_bfloat162 *)&tmp1;
            res[2] = *(__hip_bfloat162 *)&tmp2;
            res[3] = *(__hip_bfloat162 *)&tmp3;

#endif
        }
    } // end namespace DEQUANTER

    namespace SCALEHELPER
    {
        template <typename T, int VecSize>
        struct vec_scale;

        template <>
        struct vec_scale<half_t, 8>
        {
            using type = vec8_fp16;
        };

        template <>
        struct vec_scale<bhalf_t, 8>
        {
            using type = vec8_bf16;
        };

        template <>
        struct vec_scale<half_t, 4>
        {
            using type = vec4_fp16;
        };

        template <>
        struct vec_scale<bhalf_t, 4>
        {
            using type = vec4_bf16;
        };

        template <>
        struct vec_scale<half_t, 2>
        {
            using type = vec2_fp16;
        };

        template <>
        struct vec_scale<bhalf_t, 2>
        {
            using type = vec2_bf16;
        };

        template <typename T, int VecSize>
        using packed_scale = typename vec_scale<T, VecSize>::type;

    } // end namespace SCALEHELPER

} // end namespace W4A16_UTILS

using namespace W4A16_UTILS;
using namespace W4A16_UTILS::NORMAL;
using namespace W4A16_UTILS::INTERLEAVE;
using namespace W4A16_UTILS::SCALEHELPER;

#define W4A16_SCALE_SWITCH(size_k, ...) \
    [&] {                                                       \
  if (size_k == 256) {                                   \
    constexpr static int32_t PACK_SCALES = 2;                  \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    constexpr static int32_t PACK_SCALES = 4;\
    return __VA_ARGS__();                                   \
  } }();

#define W4A16_STAGE1_SCALE_SWITCH(size_k, ...) \
    [&] {                                                       \
  if (size_k == 256) {                                   \
    constexpr static int32_t PACK_SCALES = 4;                  \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    constexpr static int32_t PACK_SCALES = 8;\
    return __VA_ARGS__();                                   \
  } }();

#define W4A16_ARCH_SWITCH(arch, ...) \
    [&] {                                                       \
  if (arch != 938) {                                   \
    constexpr static bool INTERLEAVE = false;                  \
    constexpr static bool DEQUANT_16BIT = false;                  \
    return __VA_ARGS__();                                   \
  }else {                                                  \
    constexpr static bool INTERLEAVE = true;                  \
    constexpr static bool DEQUANT_16BIT = true;                  \
    return __VA_ARGS__();                                   \
  } }();

template <
    int WARP_NUM,
    int BLOCK_M, // 128
    int BLOCK_N, // 256
    int BLOCK_K, // 64
    int WARP_M,  // 64
    int WARP_N,  // 64
    int WARP_K,  // 64
    int GROUP_K, // 32
    typename Element,
    typename scalar_t2,
    bool INTERLEAVE,
    bool DEQUANT_16BIT,
    int32_t PACK_SCALES>
__forceinline__ __device__ void gemm_nt_w4a16_stage1_b128_k64_marlinS_scale_ldx(
    const Element *input_ptr,
    const uint32_t *weight_ptr,
    uint32_t *weight_zeros_ptr,
    Element *weight_scale_ptr,
    int max_n_len_offset,
    int size_n,
    floatx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)], // [1][16]
    int warp_id,
    int size_k,
    int top_k,
    const int *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int warp_m_id)
{

    using Dtype = ScalarType<Element>;
    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;

    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int warp_n_num = BLOCK_N / WARP_N; // 4
    int warp_n_id = warp_id % warp_n_num;

    const uint8_t *g_expert_qzero = reinterpret_cast<const uint8_t *>(weight_zeros_ptr);

    extern __shared__ float lds_buffer[];
    Element *lds_A = reinterpret_cast<Element *>(lds_buffer);
    Element *lds_B = reinterpret_cast<Element *>(lds_buffer);

    constexpr int32_t BLOCK_SIZE = WARP_NUM * 64;
    constexpr int32_t ELEMENTS_PER_LD_A = 128 / (sizeof(Element) * 8 /*bit*/);
    constexpr int32_t G2R_A_COLS = BLOCK_K / ELEMENTS_PER_LD_A;
    constexpr int32_t G2R_A_ROWS = BLOCK_SIZE / G2R_A_COLS;
    int32_t g2r_a_col_id = threadIdx.x % G2R_A_COLS;
    int32_t g2r_a_row_id = threadIdx.x / G2R_A_COLS;
    constexpr int32_t ELEMENTS_PER_LD_B = 128 / 4 /*bit*/;
    constexpr int32_t G2R_B_COLS = BLOCK_K / ELEMENTS_PER_LD_B;
    constexpr int32_t G2R_B_ROWS = BLOCK_SIZE / G2R_B_COLS;
    static_assert(G2R_B_ROWS == BLOCK_N, "G2R_B_ROWS != BLOCK_N");
    int32_t g2r_b_col_id = threadIdx.x % G2R_B_COLS;
    int32_t g2r_b_row_id = threadIdx.x / G2R_B_COLS;
    constexpr int32_t ELEMENTS_PER_32BIT = 32 / 4;
    constexpr int32_t INT32S_PER_128BIT = 128 / 32;
    constexpr int32_t R2S_B_COLS = BLOCK_K / ELEMENTS_PER_LD_A;

    vec8_Element<Element> a_reg[WARP_M / MFMA_M][BLOCK_K / 32];

    for (int it_out = 0; it_out < size_k; it_out += BLOCK_K * PACK_SCALES)
    {
        uint64_t scale_offset_0 = g2r_b_row_id * (size_k / GROUP_K) + g2r_b_col_id * (size_k / GROUP_K / 4 /*hard code base on BK=64 and scale marlin each 4 interleave,fix it*/) + it_out / BLOCK_K / 2 /*increase 4 each iter*/;
        uint64_t scale_offset_1 = g2r_b_row_id * (size_k / GROUP_K) + g2r_b_col_id * (size_k / GROUP_K / 4 /*hard code base on BK=64 and scale marlin each 4 interleave,fix it*/) + (size_k / GROUP_K / 2) + it_out / BLOCK_K / 2 /*increase 4 each iter*/;
        packed_scale<Element, PACK_SCALES> scale_reg;
        packed_scale<Element, PACK_SCALES / 2> *p_scale_reg = reinterpret_cast<packed_scale<Element, PACK_SCALES / 2> *>(&scale_reg);
        packed_scale<Element, PACK_SCALES / 2> *scale_p8_ptr0 = reinterpret_cast<packed_scale<Element, PACK_SCALES / 2> *>(&weight_scale_ptr[scale_offset_0]);
        packed_scale<Element, PACK_SCALES / 2> *scale_p8_ptr1 = reinterpret_cast<packed_scale<Element, PACK_SCALES / 2> *>(&weight_scale_ptr[scale_offset_1]);
        p_scale_reg[0] = scale_p8_ptr0[0];
        p_scale_reg[1] = scale_p8_ptr1[0];

#pragma unroll
        for (int it_in = 0; it_in < PACK_SCALES * BLOCK_K; it_in += BLOCK_K)
        {
            int k_start = it_out + it_in;
            vec8_Element<Element> a_val[BLOCK_M / G2R_A_ROWS];
#pragma unroll
            for (int i = 0; i < (BLOCK_M / G2R_A_ROWS); i++)
            {
                int32_t basic_row_a = g2r_a_row_id + i * G2R_A_ROWS;
                int32_t basic_col_a = g2r_a_col_id;
                int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + basic_row_a, int(sorted_token_lens - 1))];
                int global_m = std::min(sorted_token_idx / top_k, max_n_len_offset - 1);
                a_val[i] = *reinterpret_cast<const vec8_Element<Element> *>(&input_ptr[global_m * size_k + k_start + basic_col_a * ELEMENTS_PER_LD_A]);
            }
            __builtin_amdgcn_sched_barrier(0);
            int global_k_b = k_start + g2r_b_col_id * ELEMENTS_PER_LD_B;
            uint64_t b_offset = g2r_b_row_id * (size_k / ELEMENTS_PER_32BIT) + global_k_b / ELEMENTS_PER_32BIT;

            union_vec_opt_w4a16<uint32_t, INT32S_PER_128BIT> b_val;
            b_val.int4_array[0] = *reinterpret_cast<const vec<uint32_t, INT32S_PER_128BIT> *>(&weight_ptr[b_offset]);

            uint64_t scale_offset = g2r_b_row_id * (size_k / GROUP_K) + g2r_b_col_id * (size_k / GROUP_K / 2) + k_start / BLOCK_K;
            __bf16 *p_tmp = (__bf16 *)&scale_reg;
            int32_t scale_iter = it_in / BLOCK_K;
            scale_iter = (scale_iter % 2) * (PACK_SCALES / 2) + (scale_iter / 2);
            Element f_scale = scale_reg[it_in / BLOCK_K];
            __bf16 bf_tmp = p_tmp[scale_iter];
            using __bf16x2 = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;
            __bf16x2 bf2_tmp = {bf_tmp, bf_tmp};

            union scale_cvter
            {
                float f_data;
                uint32_t i_data;
            };

            scale_cvter scale_tmp;
            scale_tmp.i_data = scale_reg[scale_iter];
            scale_tmp.i_data = scale_tmp.i_data << 16;
            vec2_fp32 scale2 = {scale_tmp.f_data, scale_tmp.f_data};

            uint64_t zp_offset = (g2r_b_row_id / 2) * (size_k / GROUP_K) + global_k_b / GROUP_K;
            uint8_t zp_byte = g_expert_qzero[zp_offset];
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int i = 0; i < (BLOCK_M / G2R_A_ROWS); i++)
            {
                int32_t basic_row_a = g2r_a_row_id + i * G2R_A_ROWS;
                int32_t basic_col_a = g2r_a_col_id;
                int swizzled_col_a = (basic_row_a + basic_col_a) % G2R_A_COLS;
                if constexpr (!DEQUANT_16BIT)
                {
                    *reinterpret_cast<vec8_Element<Element> *>(&lds_A[basic_row_a * BLOCK_K + swizzled_col_a * ELEMENTS_PER_LD_A]) = a_val[i];
                }
                else
                {
                    vec8_Element<Element> tmp;
                    tmp[0] = a_val[i][0];
                    tmp[1] = a_val[i][4];
                    tmp[2] = a_val[i][1];
                    tmp[3] = a_val[i][5];
                    tmp[4] = a_val[i][2];
                    tmp[5] = a_val[i][6];
                    tmp[6] = a_val[i][3];
                    tmp[7] = a_val[i][7];
                    *reinterpret_cast<vec8_Element<Element> *>(&lds_A[basic_row_a * BLOCK_K + swizzled_col_a * ELEMENTS_PER_LD_A]) = tmp;
                }
            }
            __syncthreads();
            for (int k_step = 0; k_step < BLOCK_K; k_step += 32 /*4 lane * 8 bf16 per lane*/)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
                    int m_idx = warp_m_id * WARP_M + m_tile * MFMA_M + row_id;
                    int k_idx = k_step / 8 /*cols*/ + col_id;
                    int swizzled_col_a = (m_idx + k_idx) % G2R_A_COLS;
                    a_reg[m_tile][k_step / 32] = *reinterpret_cast<vec8_Element<Element> *>(&lds_A[m_idx * BLOCK_K + swizzled_col_a * ELEMENTS_PER_LD_A]);
                }
            }

            scalar_t2 scale_f2 = Dtype::num2num2(f_scale);

            int shift = ((g2r_b_row_id) & 1) * 4;
            constexpr uint32_t zp_val = 8; //(zp_byte >> shift) & 0x0F;

            scalar_t2 qzero_f2;
            if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
            {
                uint32_t tmp = zp_val ^ (zp_val << 16);
                static constexpr uint32_t MASK = 0x000f000f;
                static constexpr uint32_t EX = 0xC300C300;
                tmp = (tmp & MASK | EX);
                qzero_f2 = *(scalar_t2 *)&tmp;
            }
            else
            {
                qzero_f2 = Dtype::num2num2(Dtype::int2num(zp_val));
            }
            __syncthreads();
            auto dequant_gfx936 = [&](uint32_t q_weight, uint32_t *B)
            {
                union dequant_data
                {
                    vec2_fp32 f_data;
                    vec2_uint i_data;
                };
                vec2_fp32 zero2 = {-8.f * scale2[0], -8.f * scale2[1]};
                uint32_t lo = q_weight & 0x0f0f0f0f;
                uint32_t hi = (q_weight >> 4) & 0x0f0f0f0f;
                dequant_data p0, p1, p2, p3;
                asm volatile("v_cvt_f32_ubyte0 %0, %1" : "=v"(p0.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte0 %0, %1" : "=v"(p0.f_data[1]) : "v"(hi));
                asm volatile("v_cvt_f32_ubyte1 %0, %1" : "=v"(p1.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte1 %0, %1" : "=v"(p1.f_data[1]) : "v"(hi));
                asm volatile("v_cvt_f32_ubyte2 %0, %1" : "=v"(p2.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte2 %0, %1" : "=v"(p2.f_data[1]) : "v"(hi));
                asm volatile("v_cvt_f32_ubyte3 %0, %1" : "=v"(p3.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte3 %0, %1" : "=v"(p3.f_data[1]) : "v"(hi));

                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p0.f_data) : "v"(p0.f_data), "v"(scale2), "v"(zero2));
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p1.f_data) : "v"(p1.f_data), "v"(scale2), "v"(zero2));
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p2.f_data) : "v"(p2.f_data), "v"(scale2), "v"(zero2));
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p3.f_data) : "v"(p3.f_data), "v"(scale2), "v"(zero2));

                B[0] = (p0.i_data[1] & 0xffff0000) | (p0.i_data[0] >> 16);
                B[1] = (p1.i_data[1] & 0xffff0000) | (p1.i_data[0] >> 16);
                B[2] = (p2.i_data[1] & 0xffff0000) | (p2.i_data[0] >> 16);
                B[3] = (p3.i_data[1] & 0xffff0000) | (p3.i_data[0] >> 16);
            };

            reg_bf16_fp16<scalar_t2> deq_res[INT32S_PER_128BIT];
#pragma unroll
            for (int i = 0; i < INT32S_PER_128BIT; i++)
            {
                if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
                {

                    if constexpr (DEQUANT_16BIT)
                    {
                        W4A16_UTILS::DEQUANTER::dequant_weight_debug(b_val.uint_array[i], bf2_tmp, qzero_f2, deq_res[i].bf162);
                    }
                    else
                    {
                        uint32_t *B = reinterpret_cast<uint32_t *>(deq_res[i].bf162);
                        dequant_gfx936(b_val.uint_array[i], B);
                    }
                }
                else
                {
                    dequant<scalar_t2, 4>(b_val.uint_array[i], deq_res[i].bf162);
#pragma unroll
                    for (int pack_id = 0; pack_id < 4; pack_id++)
                    {
                        deq_res[i].bf162[pack_id] = __hmul2(__hsub2(deq_res[i].bf162[pack_id], qzero_f2), scale_f2);
                    }
                }

                const vec8_Element<Element> *b_shuffled = reinterpret_cast<const vec8_Element<Element> *>(deq_res[i].bf162);
                int c = g2r_b_col_id * (ELEMENTS_PER_LD_B / ELEMENTS_PER_32BIT) + i; // 列索引
                int r = g2r_b_row_id;
                int swizzled_col_b = (r + c) % R2S_B_COLS;
                *reinterpret_cast<vec8_Element<Element> *>(&lds_B[g2r_b_row_id * BLOCK_K + swizzled_col_b * ELEMENTS_PER_32BIT]) = *b_shuffled;
            }

            __syncthreads();

            for (int k_step = 0; k_step < BLOCK_K; k_step += 32 /*4 lane * 8 bf16 per lane*/)
            {

                vec8_Element<Element> b_reg[WARP_N / MFMA_N];
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
                    int n_idx = warp_n_id * WARP_N + n_tile * MFMA_N + row_id;
                    int k_idx = k_step / 8 /*cols*/ + col_id;
                    int swizzled_col_b = (n_idx + k_idx) % R2S_B_COLS;
                    b_reg[n_tile] = *reinterpret_cast<vec8_Element<Element> *>(&lds_B[n_idx * BLOCK_K + swizzled_col_b * ELEMENTS_PER_LD_A]);
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {

                        vec4_Element<Element> *a_vec4 = reinterpret_cast<vec4_Element<Element> *>(&a_reg[m_tile][k_step / 32]);
                        vec4_Element<Element> *b_vec4 = reinterpret_cast<vec4_Element<Element> *>(&b_reg[n_tile]);
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[0], b_vec4[0], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[1], b_vec4[1], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <bool Is_store_A,
          int A_prefetch_level,
          int B_prefetch_level,
          int WARP_NUM,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGE,
          int GROUP_N,
          int GROUP_K,
          typename Element,
          typename scalar_t2,
          bool INTERLEAVE,
          bool DEQUANT_16BIT,
          int32_t SCALE_CO_LOAD_COUNT>
__forceinline__ __device__ void gemm_nt_w4a16_stage_lambda_scale_ldx(
    const Element *input_ptr,
    const uint32_t *weight_ptr,
    uint32_t *weight_zeros_ptr,
    Element *weight_scale_ptr,
    int max_n_len_offset,
    int size_n,
    union_vec_opt_w4a16_A<Element, WARP_K / 4> A_reg[][STAGE],     // 2 = stage
    union_vec_opt_w4a16<uint32_t, WARP_K / 32> B_int_reg[][STAGE], // 2 = stage
    Element B_scale_reg[][STAGE],
    uint32_t B_zeros_reg[][STAGE],
    reg_bf16_fp16<scalar_t2> B_reg[][STAGE][WARP_K / 32], // 2 = stage
    floatx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k: 256 or 512 or 1024 or 2048
    int seqlen_B_stride, // size_k: 7168
    int top_k,           // 8
    const int *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    const int warp_m_id)
{

    const int size_k = seqlen_A_stride;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K_A = 32;  // k方向 4个线程 每个线程读16个int8
    constexpr int READ_K_B = 128; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_m_num = BLOCK_M / WARP_M;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_nk_id = warp_id % (warp_n_num * warp_k_num);
    int warp_k_id = warp_nk_id % warp_k_num;
    int warp_n_id = warp_nk_id / warp_k_num;

    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K;
    constexpr int32_t IL_COUNT = 4;
    constexpr int32_t SCALE_RANGE = 32;
    int k_start_scale_zeros = (col_id % IL_COUNT) * (size_k / SCALE_RANGE / IL_COUNT) + (col_id / IL_COUNT);

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K;
    const int stage_offset_scale_zeros = SCALE_CO_LOAD_COUNT;

    auto g_expert_scales = tcp_cache_swizzle_func<1, Element>(weight_scale_ptr);
    const vec8_Element<Element> *g_a = reinterpret_cast<const vec8_Element<Element> *>(input_ptr);

    auto g_expert_qzero = tcp_cache_swizzle_func<1, uint8_t>(reinterpret_cast<const uint8_t *>(weight_zeros_ptr));

    int g_row_B[WARP_N / MFMA_N];
    union_vec<uint32_t, WARP_N / MFMA_N> expert_qzeros_groups;
    union_vec<Element, WARP_N / MFMA_N> expert_scales_groups;

    using Dtype = ScalarType<Element>;

    scalar_t2 scale_f2[WARP_N / MFMA_N];
    scalar_t2 qzero_f2[WARP_N / MFMA_N];

    packed_scale<Element, SCALE_CO_LOAD_COUNT> scale_reg[WARP_N / MFMA_N];

    constexpr int32_t ElEMENTS = 128 / (8 * sizeof(Element));
    constexpr int32_t A_BASIC_COLS = BLOCK_K / ElEMENTS;
    constexpr int32_t A_BASIC_ROWS = WARP_NUM * 64 / A_BASIC_COLS;
    constexpr int32_t IT_LDS_STRIDE = A_BASIC_ROWS * A_BASIC_COLS;
    constexpr int32_t G2S_LDA_COUNT = BLOCK_M / A_BASIC_ROWS;
    int32_t basic_col_id = threadIdx.x % A_BASIC_COLS;
    int32_t basic_row_id = threadIdx.x / A_BASIC_COLS;
    int g_row_A[G2S_LDA_COUNT] /*bytes*/ = {0};

    extern __shared__ float lds_buffer[]; /*BM * BK */
    vec8_Element<Element> g2r_a[STAGE][G2S_LDA_COUNT] = {0};
    vec8_Element<Element> *buffer_a[STAGE];
    for (int32_t i = 0; i < STAGE; i++)
    {
        buffer_a[i] = reinterpret_cast<vec8_Element<Element> *>(lds_buffer + i * BLOCK_M * BLOCK_K / sizeof(float) * sizeof(Element));
    }

#pragma unroll
    for (int it_g2s = 0; it_g2s < G2S_LDA_COUNT; it_g2s++)
    {
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + it_g2s * A_BASIC_ROWS + basic_row_id, int(sorted_token_lens - 1))];
        g_row_A[it_g2s] = ((std::min(sorted_token_idx / top_k, max_n_len_offset - 1)) * size_k + basic_col_id * ElEMENTS) / ElEMENTS;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = ((warp_n_id * WARP_N + n_tile * MFMA_N + row_id) * size_k + col_id * 32) / 2;
    }

    auto load_scale = [&]()
    {
#pragma unroll
        for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
        {
            int scale_offset_n = (warp_n_id * WARP_N + row_id + (n_tile)*MFMA_N) * size_k / SCALE_RANGE;
            int scales_offset = scale_offset_n + k_start_scale_zeros;
            if constexpr (SCALE_CO_LOAD_COUNT == 4)
            {
                W4A16_UTILS::DATALOADER::buffer_load_reg_dwordx2_half4(weight_scale_ptr, scale_reg[n_tile], 0, scales_offset * sizeof(Element));
            }
            else
            {
                W4A16_UTILS::DATALOADER::buffer_load_reg_dword_half2(weight_scale_ptr, scale_reg[n_tile], 0, scales_offset * sizeof(Element));
            }
        }
    };

    auto advance_scale_offset = [&]()
    {
        k_start_scale_zeros += stage_offset_scale_zeros;
    };

    auto g2r = [&](int32_t it_stage)
    {
#pragma unroll
        for (int it_g2s_a = 0; it_g2s_a < G2S_LDA_COUNT; it_g2s_a++)
        {
            W4A16_UTILS::DATALOADER::buffer_load_reg_dwordx4_half8(input_ptr, g2r_a[it_stage][it_g2s_a], 0, (g_row_A[it_g2s_a] + k_start / ElEMENTS) * 16);
        }
#pragma unroll
        for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K_B; k_tile++)
            {
                buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[(n_tile)][it_stage].int4_array[k_tile], 0, g_row_B[(n_tile)] + k_tile * 128 / 2 + k_start_b / 2);
            }
        }
    };

    auto advance_AB_offset = [&]()
    {
        k_start += stage_offset;
        k_start_b += stage_offset_b;
    };

    auto matrixA_r2s = [&](int32_t it_stage)
    {
#pragma unroll
        for (int it_g2s_a = 0; it_g2s_a < G2S_LDA_COUNT; it_g2s_a++)
        {
            int32_t lds_row = basic_row_id + it_g2s_a * A_BASIC_ROWS;
            int32_t lds_col = (basic_col_id + lds_row) % A_BASIC_COLS;
            int32_t lds_offset = lds_row * A_BASIC_COLS + lds_col;
            if constexpr (DEQUANT_16BIT | !std::is_same_v<Element, __hip_bfloat16>)
            {
                vec8_Element<Element> tmp;
                tmp[0] = g2r_a[it_stage][it_g2s_a][0];
                tmp[1] = g2r_a[it_stage][it_g2s_a][4];
                tmp[2] = g2r_a[it_stage][it_g2s_a][1];
                tmp[3] = g2r_a[it_stage][it_g2s_a][5];
                tmp[4] = g2r_a[it_stage][it_g2s_a][2];
                tmp[5] = g2r_a[it_stage][it_g2s_a][6];
                tmp[6] = g2r_a[it_stage][it_g2s_a][3];
                tmp[7] = g2r_a[it_stage][it_g2s_a][7];
                buffer_a[0][lds_offset] = tmp;
            }
            else
            {
                buffer_a[0][lds_offset] = g2r_a[it_stage][it_g2s_a];
            }
        }
    };

    auto matrix_s2r = [&](int32_t it_stage)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K_A; k_tile++)
            {
                int row = warp_m_id * WARP_M + m_tile * MFMA_M + row_id;
                int col = (row + (k_tile + col_id * (WARP_K / READ_K_A))) % A_BASIC_COLS;
                A_reg[m_tile][it_stage].input_half8[k_tile] = buffer_a[0][row * A_BASIC_COLS + col];
            }
        }
    };

    auto dequantfp16 = [&](int32_t it_stage, int32_t it_scale_stage)
    {
#pragma unroll
        for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
        {
            qzero_f2[n_tile] = Dtype::num2num2(Dtype::int2num(8));
            scale_f2[n_tile] = Dtype::num2num2(scale_reg[n_tile][it_scale_stage]);
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                dequant<scalar_t2, 4>(B_int_reg[n_tile][it_stage].uint_array[j], B_reg[n_tile][it_stage][j].bf162);
            }

#pragma unroll
            for (int j = 0; j < 4; j++)
            {
#pragma unroll
                for (int pack_id = 0; pack_id < 4; pack_id++)
                {
                    B_reg[n_tile][it_stage][j].bf162[pack_id] = __hmul2(__hsub2(B_reg[n_tile][it_stage][j].bf162[pack_id], qzero_f2[n_tile]), scale_f2[n_tile]);
                }
            }
        }
    };

    auto dequantfp32 = [&](int32_t it_stage, int32_t it_scale_stage)
    {
#pragma unroll
        for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
        {
            union scale_cvter
            {
                float f_data;
                uint32_t i_data;
            };

            scale_cvter scale_tmp;
            scale_tmp.i_data = scale_reg[n_tile][it_scale_stage]; // static_cast<uint32_t>(scale_reg[n_tile][it_scale_stage]);
            scale_tmp.i_data = scale_tmp.i_data << 16;
            vec2_fp32 scale2 = {scale_tmp.f_data, scale_tmp.f_data};
            auto dequant_gfx936 = [&](uint32_t q_weight, uint32_t *B)
            {
                union dequant_data
                {
                    vec2_fp32 f_data;
                    vec2_uint i_data;
                };
                vec2_fp32 zero2 = {-8.f * scale2[0], -8.f * scale2[1]};
                uint32_t lo = q_weight & 0x0f0f0f0f;
                uint32_t hi = (q_weight >> 4) & 0x0f0f0f0f;
                dequant_data p0, p1, p2, p3;
                asm volatile("v_cvt_f32_ubyte0 %0, %1" : "=v"(p0.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte0 %0, %1" : "=v"(p0.f_data[1]) : "v"(hi));
                asm volatile("v_cvt_f32_ubyte1 %0, %1" : "=v"(p1.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte1 %0, %1" : "=v"(p1.f_data[1]) : "v"(hi));
                asm volatile("v_cvt_f32_ubyte2 %0, %1" : "=v"(p2.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte2 %0, %1" : "=v"(p2.f_data[1]) : "v"(hi));
                asm volatile("v_cvt_f32_ubyte3 %0, %1" : "=v"(p3.f_data[0]) : "v"(lo));
                asm volatile("v_cvt_f32_ubyte3 %0, %1" : "=v"(p3.f_data[1]) : "v"(hi));

                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p0.f_data) : "v"(p0.f_data), "v"(scale2), "v"(zero2));
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p1.f_data) : "v"(p1.f_data), "v"(scale2), "v"(zero2));
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p2.f_data) : "v"(p2.f_data), "v"(scale2), "v"(zero2));
                asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p3.f_data) : "v"(p3.f_data), "v"(scale2), "v"(zero2));

                B[0] = (p0.i_data[1] & 0xffff0000) | (p0.i_data[0] >> 16);
                B[1] = (p1.i_data[1] & 0xffff0000) | (p1.i_data[0] >> 16);
                B[2] = (p2.i_data[1] & 0xffff0000) | (p2.i_data[0] >> 16);
                B[3] = (p3.i_data[1] & 0xffff0000) | (p3.i_data[0] >> 16);
            };
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                uint32_t *B = reinterpret_cast<uint32_t *>(B_reg[n_tile][it_stage][j].bf162);
                dequant_gfx936(B_int_reg[n_tile][it_stage].uint_array[j], B);
            }
        }
    };

    auto matmul = [&](int32_t it_stage)
    {
#pragma unroll
        for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                {
                    vec4_Element<Element> *a_vec4 = reinterpret_cast<vec4_Element<Element> *>(&(A_reg[m_tile][it_stage].input_half[k_tile * 8]));
                    vec4_Element<Element> *b_vec4 = reinterpret_cast<vec4_Element<Element> *>(&(B_reg[n_tile][it_stage][k_tile].vec2_i16[0]));

                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[0], b_vec4[0], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[1], b_vec4[1], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                }
            }
        }
    };

    auto dequantfp32_and_matmul = [&](int32_t it_stage, int32_t it_scale_stage)
    {

#pragma unroll
        for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
        {
            union scale_cvter
            {
                float f_data;
                uint32_t i_data;
            };

            scale_cvter scale_tmp;
            scale_tmp.i_data = scale_reg[n_tile][it_scale_stage]; // static_cast<uint32_t>(scale_reg[n_tile][it_scale_stage]);
            scale_tmp.i_data = scale_tmp.i_data << 16;
            vec2_fp32 scale2 = {scale_tmp.f_data, scale_tmp.f_data};
            if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
            {
                uint32_t tmp = 0x80008;
                static constexpr uint32_t MASK = 0x000f000f;
                static constexpr uint32_t EX = 0xC300C300;
                tmp = (tmp & MASK | EX);
                qzero_f2[n_tile] = *(scalar_t2 *)&tmp;
            }
            else
            {
                qzero_f2[n_tile] = Dtype::num2num2(Dtype::int2num(8));
            }
            if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
            {
                auto dequant_gfx936 = [&](uint32_t q_weight, uint32_t *B)
                {
                    union dequant_data
                    {
                        vec2_fp32 f_data;
                        vec2_uint i_data;
                    };
                    vec2_fp32 zero2 = {-8.f * scale2[0], -8.f * scale2[1]};
                    uint32_t lo = q_weight & 0x0f0f0f0f;
                    uint32_t hi = (q_weight >> 4) & 0x0f0f0f0f;
                    dequant_data p0, p1, p2, p3;
                    asm volatile("v_cvt_f32_ubyte0 %0, %1" : "=v"(p0.f_data[0]) : "v"(lo));
                    asm volatile("v_cvt_f32_ubyte0 %0, %1" : "=v"(p0.f_data[1]) : "v"(hi));
                    asm volatile("v_cvt_f32_ubyte1 %0, %1" : "=v"(p1.f_data[0]) : "v"(lo));
                    asm volatile("v_cvt_f32_ubyte1 %0, %1" : "=v"(p1.f_data[1]) : "v"(hi));
                    asm volatile("v_cvt_f32_ubyte2 %0, %1" : "=v"(p2.f_data[0]) : "v"(lo));
                    asm volatile("v_cvt_f32_ubyte2 %0, %1" : "=v"(p2.f_data[1]) : "v"(hi));
                    asm volatile("v_cvt_f32_ubyte3 %0, %1" : "=v"(p3.f_data[0]) : "v"(lo));
                    asm volatile("v_cvt_f32_ubyte3 %0, %1" : "=v"(p3.f_data[1]) : "v"(hi));

                    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p0.f_data) : "v"(p0.f_data), "v"(scale2), "v"(zero2));
                    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p1.f_data) : "v"(p1.f_data), "v"(scale2), "v"(zero2));
                    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p2.f_data) : "v"(p2.f_data), "v"(scale2), "v"(zero2));
                    asm volatile("v_pk_fma_f32 %0, %1, %2, %3" : "=v"(p3.f_data) : "v"(p3.f_data), "v"(scale2), "v"(zero2));

                    B[0] = (p0.i_data[1] & 0xffff0000) | (p0.i_data[0] >> 16);
                    B[1] = (p1.i_data[1] & 0xffff0000) | (p1.i_data[0] >> 16);
                    B[2] = (p2.i_data[1] & 0xffff0000) | (p2.i_data[0] >> 16);
                    B[3] = (p3.i_data[1] & 0xffff0000) | (p3.i_data[0] >> 16);
                };
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    uint32_t *B = reinterpret_cast<uint32_t *>(B_reg[n_tile][it_stage][j].bf162);
                    dequant_gfx936(B_int_reg[n_tile][it_stage].uint_array[j], B);
                }
            }
            else
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    dequant<scalar_t2, 4>(B_int_reg[n_tile][it_stage].uint_array[j], B_reg[n_tile][it_stage][j].bf162);
                }
            }
            if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
            {
            }
            else
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
#pragma unroll
                    for (int pack_id = 0; pack_id < 4; pack_id++)
                    {
                        B_reg[n_tile][it_stage][j].bf162[pack_id] = __hmul2(__hsub2(B_reg[n_tile][it_stage][j].bf162[pack_id], qzero_f2[n_tile]), scale_f2[n_tile]);
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                {
                    vec4_Element<Element> *a_vec4 = reinterpret_cast<vec4_Element<Element> *>(&(A_reg[m_tile][it_stage].input_half[k_tile * 8]));
                    vec4_Element<Element> *b_vec4 = reinterpret_cast<vec4_Element<Element> *>(&(B_reg[n_tile][it_stage][k_tile].vec2_i16[0]));

                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[0], b_vec4[0], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[1], b_vec4[1], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                }
            }
        }
    };

    auto dequantbf16_and_matmul = [&](int32_t it_stage, int32_t it_scale_stage)
    {

#pragma unroll
        for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
        {
            __bf16 *p_tmp = (__bf16 *)&(scale_reg[n_tile]);
            __bf16 bf_tmp = p_tmp[it_scale_stage];
            using __bf16x2 = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;
            __bf16x2 bf2_tmp = {bf_tmp, bf_tmp};
            if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
            {
                uint32_t tmp = 0x80008;
                static constexpr uint32_t MASK = 0x000f000f;
                static constexpr uint32_t EX = 0xC300C300;
                tmp = (tmp & MASK | EX);
                qzero_f2[n_tile] = *(scalar_t2 *)&tmp;
            }
            else
            {
                qzero_f2[n_tile] = Dtype::num2num2(Dtype::int2num(expert_qzeros_groups.uint32_array[n_tile]));
            }
            if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    W4A16_UTILS::DEQUANTER::dequant_weight_debug(B_int_reg[n_tile][it_stage].uint_array[j], bf2_tmp, qzero_f2[n_tile], B_reg[n_tile][it_stage][j].bf162);
                }
            }
            else
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    dequant<scalar_t2, 4>(B_int_reg[n_tile][it_stage].uint_array[j], B_reg[n_tile][it_stage][j].bf162);
                }
            }
            if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
            {
            }
            else
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
#pragma unroll
                    for (int pack_id = 0; pack_id < 4; pack_id++)
                    {
                        B_reg[n_tile][it_stage][j].bf162[pack_id] = __hmul2(__hsub2(B_reg[n_tile][it_stage][j].bf162[pack_id], qzero_f2[n_tile]), scale_f2[n_tile]);
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                {
                    vec4_Element<Element> *a_vec4 = reinterpret_cast<vec4_Element<Element> *>(&(A_reg[m_tile][it_stage].input_half[k_tile * 8]));
                    vec4_Element<Element> *b_vec4 = reinterpret_cast<vec4_Element<Element> *>(&(B_reg[n_tile][it_stage][k_tile].vec2_i16[0]));
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[0], b_vec4[0], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = W4A16_UTILS::mmac<Element, INTERLEAVE>(a_vec4[1], b_vec4[1], C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                }
            }
        }
    };

    auto dequant_and_matmul = [&](int32_t it_stage, int32_t it_scale_stage)
    {
        if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
        {
            if constexpr (DEQUANT_16BIT)
            {
                auto dequantbf16 = [&](int32_t it_stage, int32_t it_scale_stage)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < (WARP_N / MFMA_N); n_tile++)
                    {
                        __bf16 *p_tmp = (__bf16 *)&(scale_reg[n_tile]);
                        __bf16 bf_tmp = p_tmp[it_scale_stage];
                        using __bf16x2 = __attribute__((__vector_size__(2 * sizeof(__bf16)))) __bf16;
                        __bf16x2 bf2_tmp = {bf_tmp, bf_tmp};
                        uint32_t tmp = 0x80008;
                        static constexpr uint32_t MASK = 0x000f000f;
                        static constexpr uint32_t EX = 0xC300C300;
                        tmp = (tmp & MASK | EX);
                        qzero_f2[n_tile] = *(scalar_t2 *)&tmp;
#pragma unroll
                        for (int j = 0; j < 4; j++)
                        {
                            W4A16_UTILS::DEQUANTER::dequant_weight_debug(B_int_reg[n_tile][it_stage].uint_array[j], bf2_tmp, qzero_f2[n_tile], B_reg[n_tile][it_stage][j].bf162);
                        }
                    }
                };

                dequantbf16(it_stage, it_scale_stage);
            }
            else
            {
                dequantfp32(it_stage, it_scale_stage);
            }
        }
        else
        {
            dequantfp16(it_stage, it_scale_stage);
        }
        matmul(it_stage);
    };

    auto first_stage = [&](int32_t it_stage)
    {
        const int32_t next_stage = (it_stage + 1) % STAGE;
        const int32_t cur_stage = it_stage % STAGE;
        g2r(next_stage);
        vmcnt_wait_w4a16(((WARP_N / MFMA_N) * (WARP_K / READ_K_B) + G2S_LDA_COUNT) + (WARP_N / MFMA_N) * (WARP_K / READ_K_B) * 2);
        matrixA_r2s(cur_stage);
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        matrix_s2r(cur_stage);
        vmcnt_wait_w4a16((WARP_N / MFMA_N) * (WARP_K / READ_K_B) + G2S_LDA_COUNT);
        dequant_and_matmul(cur_stage, it_stage);
        advance_AB_offset();
    };

    auto mid_stage = [&](int32_t it_stage)
    {
        const int32_t next_stage = (it_stage + 1) % STAGE;
        const int32_t cur_stage = it_stage % STAGE;
        g2r(next_stage);
        vmcnt_wait_w4a16(((WARP_N / MFMA_N) * (WARP_K / READ_K_B) * 2 + G2S_LDA_COUNT));
        matrixA_r2s(cur_stage);
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        matrix_s2r(cur_stage);
        vmcnt_wait_w4a16((WARP_N / MFMA_N) * (WARP_K / READ_K_B) + G2S_LDA_COUNT);
        dequant_and_matmul(cur_stage, it_stage);
        advance_AB_offset();
    };

    auto last_stage = [&](int32_t it_stage)
    {
        const int32_t next_stage = (it_stage + 1) % STAGE;
        const int32_t cur_stage = it_stage % STAGE;
        g2r(next_stage);
        vmcnt_wait_w4a16(((WARP_N / MFMA_N) * (WARP_K / READ_K_B) * 2 + G2S_LDA_COUNT));
        matrixA_r2s(cur_stage);
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        matrix_s2r(cur_stage);
        vmcnt_wait_w4a16((WARP_N / MFMA_N) * (WARP_K / READ_K_B) + G2S_LDA_COUNT);
        dequant_and_matmul(cur_stage, it_stage);
        load_scale();
        advance_scale_offset();
        advance_AB_offset();
    };

    auto remainder_stage = [&](int32_t it_stage)
    {
        const int32_t cur_stage = it_stage % STAGE;
        vmcnt_wait_w4a16((WARP_N / MFMA_N) * (WARP_K / READ_K_B));
        matrixA_r2s(cur_stage);
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        asm volatile("s_nop 3 \n\t");
        matrix_s2r(cur_stage);
        vmcnt_wait_w4a16(0);
        dequant_and_matmul(cur_stage, it_stage);
    };

#pragma unroll
    for (int it_stage = 0; it_stage < 1; it_stage++)
    {
        g2r(it_stage);
        load_scale();
        advance_AB_offset();
        advance_scale_offset();
    }
    __builtin_amdgcn_sched_barrier(0);

    for (int it_k = 0; it_k < (size_k / (WARP_K * SCALE_CO_LOAD_COUNT) - 1 /*leave one stage remainder*/); it_k++)
    {
        first_stage(0);
#pragma unroll
        for (int it_stage = 1; it_stage < (SCALE_CO_LOAD_COUNT - 1); it_stage++)
        {
            mid_stage(it_stage);
        } // end it_stage
        last_stage(SCALE_CO_LOAD_COUNT - 1);
    } // end it_k

    first_stage(0);
#pragma unroll
    for (int it_stage = 1; it_stage < (SCALE_CO_LOAD_COUNT - 1); it_stage++)
    {
        mid_stage(it_stage);
    } // end it_stage
    remainder_stage(SCALE_CO_LOAD_COUNT - 1);

    extern __shared__ float out_smem_w4a16[]; // 声明lds信息
    if constexpr (warp_k_num > 1)
    {
        constexpr int pading_n = WARP_N + 1;

        if (warp_k_id > 0)
        { // 0和1 只需要 warpk_id为1去拷贝数据
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
                    *(floatx4 *)(&out_smem_w4a16[(warp_m_id * WARP_M + m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
                }
            }
        }

        __syncthreads();

        if (warp_k_id == 0)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < warp_k_num - 1; k_tile++)
                    {
                        floatx4 temp = *(floatx4 *)(&out_smem_w4a16[(warp_m_id * WARP_M + m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile][i] += temp[i];
                        }
                    }
                }
            }
        }
    }
}
