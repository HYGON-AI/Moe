// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT


#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

#include "intrinsic_2.h"
#include "intrinsic.h"
#include "moe_wna16_utils.h"

template <typename T>
__forceinline__ __device__ uint16_t num_as_u16_bits(T value)
{
    static_assert(sizeof(T) == sizeof(uint16_t));
    uint16_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename scalar_t2>
__forceinline__ __device__ void dequant_w8a16_to_reg(
    uint32_t packed0,
    uint32_t packed1,
    reg_bf16_fp16<scalar_t2> &dst)
{
    if constexpr (std::is_same_v<scalar_t2, __hip_bfloat162>)
    {
        uint32_t packed[2] = {packed0 ^ 0x80808080u, packed1 ^ 0x80808080u};
        float fp32_intermediates[4];
        uint32_t *fp32_bits = reinterpret_cast<uint32_t *>(fp32_intermediates);
        uint32_t *bf16_result = reinterpret_cast<uint32_t *>(dst.bf162);
        static constexpr uint32_t fp32_base = 0x4B000000;
        static constexpr float signed_i8_offset = 8388736.f; // 2^23 + 128

#pragma unroll
        for (int pack_id = 0; pack_id < 2; pack_id++)
        {
            fp32_bits[0] = __byte_perm(packed[pack_id], fp32_base, 0x7650);
            fp32_bits[1] = __byte_perm(packed[pack_id], fp32_base, 0x7651);
            fp32_bits[2] = __byte_perm(packed[pack_id], fp32_base, 0x7652);
            fp32_bits[3] = __byte_perm(packed[pack_id], fp32_base, 0x7653);

            fp32_intermediates[0] -= signed_i8_offset;
            fp32_intermediates[1] -= signed_i8_offset;
            fp32_intermediates[2] -= signed_i8_offset;
            fp32_intermediates[3] -= signed_i8_offset;

            bf16_result[pack_id * 2 + 0] = __byte_perm(fp32_bits[0], fp32_bits[1], 0x7632);
            bf16_result[pack_id * 2 + 1] = __byte_perm(fp32_bits[2], fp32_bits[3], 0x7632);
        }
    }
    else
    {
        uint32_t packed[2] = {packed0, packed1};
#pragma unroll
        for (int pack_id = 0; pack_id < 2; pack_id++)
        {
            const uint8_t u0 = (packed[pack_id] >> 0) & 0xFF;
            const uint8_t u1 = (packed[pack_id] >> 8) & 0xFF;
            const uint8_t u2 = (packed[pack_id] >> 16) & 0xFF;
            const uint8_t u3 = (packed[pack_id] >> 24) & 0xFF;
            const int8_t v0 = (u0 < 128) ? static_cast<int8_t>(u0) : static_cast<int8_t>(u0 - 256);
            const int8_t v1 = (u1 < 128) ? static_cast<int8_t>(u1) : static_cast<int8_t>(u1 - 256);
            const int8_t v2 = (u2 < 128) ? static_cast<int8_t>(u2) : static_cast<int8_t>(u2 - 256);
            const int8_t v3 = (u3 < 128) ? static_cast<int8_t>(u3) : static_cast<int8_t>(u3 - 256);
            dst.bf162[pack_id * 2 + 0] = __halves2half2(__float2half(static_cast<float>(v0)),
                                                        __float2half(static_cast<float>(v1)));
            dst.bf162[pack_id * 2 + 1] = __halves2half2(__float2half(static_cast<float>(v2)),
                                                        __float2half(static_cast<float>(v3)));
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
          int SIZE_K,
          typename Element,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_w8a16(
    const Element *input_ptr,
    const uint32_t *weight_ptr,
    Element *weight_scale_ptr,
    int max_n_len_offset,
    int size_n,
    union_vec_opt_w8a16_A<Element, WARP_K / 4> A_reg[][STAGE],     // 2 = stage
    union_vec_opt_w8a16<uint32_t, WARP_K / 16> B_int_reg[][STAGE], // 2 = stage
    Element B_scale_reg[][STAGE],
    reg_bf16_fp16<typename ScalarType<Element>::scalar_t2> B_reg[][STAGE][WARP_K / 32], // 2 = stage
    floatx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int top_k,           // 8
    const int *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx)
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
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    using Dtype = ScalarType<Element>;
    using scalar_t2 = typename ScalarType<Element>::scalar_t2;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = ((std::min(sorted_token_idx / top_k, max_n_len_offset - 1)) * size_k + col_id * 32) * 2;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * WARP_N + n_tile * MFMA_N) * 128 + row_id * 32 + col_id * 32 * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K_A; k_tile++)
            {
                buffer_load_reg_dwordx4_half8(input_ptr, A_reg[m_tile][i].input_half8[k_tile], 0, g_row_A[m_tile] + k_tile * 8 * 2 + k_start * 2);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K_B; k_tile++)
            {
                int base_offset = g_row_B[n_tile] + k_tile * 128 * size_n + k_start_b;
                buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][i].int4_array[k_tile * 2], 0, base_offset);
                buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][i].int4_array[k_tile * 2 + 1], 0, base_offset + 16);
            }
        }
        k_start += stage_offset;
        k_start_b += stage_offset_b;
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_barrier\n");
        __builtin_amdgcn_sched_barrier(0);
    }
    k_start -= stage_offset;
    k_start_b -= stage_offset_b;
#pragma unroll
    for (int lll = 0; lll < (SIZE_K - STAGE * stage_offset + 2 * stage_offset - 1) / (STAGE * stage_offset); lll++)
    {

#pragma unroll
        for (int i = 0; i < STAGE; ++i)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K_A; k_tile++)
                {
                    buffer_load_reg_dwordx4_half8(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].input_half8[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset * 2 + k_tile * 8 * 2 + k_start * 2);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K_B; k_tile++)
                {
                    int base_offset = g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 128 * size_n + k_start_b;
                    buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile * 2], 0, base_offset);
                    buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile * 2 + 1], 0, base_offset + 16);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K_A + WARP_N / MFMA_N * WARP_K / READ_K_B * 2) * (STAGE - 1));

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    dequant_w8a16_to_reg<scalar_t2>(
                        B_int_reg[n_tile][i].uint_array[j * 2],
                        B_int_reg[n_tile][i].uint_array[j * 2 + 1],
                        B_reg[n_tile][i][j]);
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {

                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(vec4_Element<Element>{A_reg[m_tile][i].input_half[k_tile * 8 + 0],
                                                                                                                   A_reg[m_tile][i].input_half[k_tile * 8 + 1],
                                                                                                                   A_reg[m_tile][i].input_half[k_tile * 8 + 2],
                                                                                                                   A_reg[m_tile][i].input_half[k_tile * 8 + 3]},
                                                                                             vec4_Element<Element>{B_reg[n_tile][i][k_tile].vec2_i16[0][0],
                                                                                                                   B_reg[n_tile][i][k_tile].vec2_i16[0][1],
                                                                                                                   B_reg[n_tile][i][k_tile].vec2_i16[1][0],
                                                                                                                   B_reg[n_tile][i][k_tile].vec2_i16[1][1]},
                                                                                             C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(vec4_Element<Element>{A_reg[m_tile][i].input_half[k_tile * 8 + 4],
                                                                                                                   A_reg[m_tile][i].input_half[k_tile * 8 + 5],
                                                                                                                   A_reg[m_tile][i].input_half[k_tile * 8 + 6],
                                                                                                                   A_reg[m_tile][i].input_half[k_tile * 8 + 7]},
                                                                                             vec4_Element<Element>{B_reg[n_tile][i][k_tile].vec2_i16[2][0],
                                                                                                                   B_reg[n_tile][i][k_tile].vec2_i16[2][1],
                                                                                                                   B_reg[n_tile][i][k_tile].vec2_i16[3][0],
                                                                                                                   B_reg[n_tile][i][k_tile].vec2_i16[3][1]},
                                                                                             C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        } // stage end
        k_start += stage_offset * STAGE;
        k_start_b += stage_offset_b * STAGE;
    } // k_loop end

    int i = 0;
#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K_A; k_tile++)
        {
            buffer_load_reg_dwordx4_half8(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].input_half8[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset * 2 + k_tile * 8 * 2 + k_start * 2);
        }
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K_B; k_tile++)
        {
            int base_offset = g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 128 * size_n + k_start_b;
            buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile * 2], 0, base_offset);
            buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile * 2 + 1], 0, base_offset + 16);
        }
    }

    vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K_A + WARP_N / MFMA_N * WARP_K / READ_K_B * 2) * (STAGE - 1));

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            dequant_w8a16_to_reg<scalar_t2>(
                B_int_reg[n_tile][i].uint_array[j * 2],
                B_int_reg[n_tile][i].uint_array[j * 2 + 1],
                B_reg[n_tile][i][j]);
        }
    }

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
            {

                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(vec4_Element<Element>{A_reg[m_tile][i].input_half[k_tile * 8 + 0],
                                                                                                           A_reg[m_tile][i].input_half[k_tile * 8 + 1],
                                                                                                           A_reg[m_tile][i].input_half[k_tile * 8 + 2],
                                                                                                           A_reg[m_tile][i].input_half[k_tile * 8 + 3]},
                                                                                     vec4_Element<Element>{B_reg[n_tile][i][k_tile].vec2_i16[0][0],
                                                                                                           B_reg[n_tile][i][k_tile].vec2_i16[0][1],
                                                                                                           B_reg[n_tile][i][k_tile].vec2_i16[1][0],
                                                                                                           B_reg[n_tile][i][k_tile].vec2_i16[1][1]},
                                                                                     C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(vec4_Element<Element>{A_reg[m_tile][i].input_half[k_tile * 8 + 4],
                                                                                                           A_reg[m_tile][i].input_half[k_tile * 8 + 5],
                                                                                                           A_reg[m_tile][i].input_half[k_tile * 8 + 6],
                                                                                                           A_reg[m_tile][i].input_half[k_tile * 8 + 7]},
                                                                                     vec4_Element<Element>{B_reg[n_tile][i][k_tile].vec2_i16[2][0],
                                                                                                           B_reg[n_tile][i][k_tile].vec2_i16[2][1],
                                                                                                           B_reg[n_tile][i][k_tile].vec2_i16[3][0],
                                                                                                           B_reg[n_tile][i][k_tile].vec2_i16[3][1]},
                                                                                     C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
            }
        }
    }
    i++;

#pragma unroll
    for (int ii = 1; ii < STAGE; ++ii)
    {
        vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K_A + WARP_N / MFMA_N * WARP_K / READ_K_B * 2) * (STAGE - 1 - ii));

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                const int stage_idx = (i + ii - 1) % STAGE;
                dequant_w8a16_to_reg<scalar_t2>(
                    B_int_reg[n_tile][stage_idx].uint_array[j * 2],
                    B_int_reg[n_tile][stage_idx].uint_array[j * 2 + 1],
                    B_reg[n_tile][stage_idx][j]);
            }
        }

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                {
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(vec4_Element<Element>{A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 0],
                                                                                                               A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 1],
                                                                                                               A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 2],
                                                                                                               A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 3]},
                                                                                         vec4_Element<Element>{B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[0][0],
                                                                                                               B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[0][1],
                                                                                                               B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[1][0],
                                                                                                               B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[1][1]},
                                                                                         C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(vec4_Element<Element>{A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 4],
                                                                                                               A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 5],
                                                                                                               A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 6],
                                                                                                               A_reg[m_tile][(i + ii - 1) % STAGE].input_half[k_tile * 8 + 7]},
                                                                                         vec4_Element<Element>{B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[2][0],
                                                                                                               B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[2][1],
                                                                                                               B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[3][0],
                                                                                                               B_reg[n_tile][(i + ii - 1) % STAGE][k_tile].vec2_i16[3][1]},
                                                                                         C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                }
            }
        }
    }

    extern __shared__ float out_smem_w8a16[]; // 声明lds信息
#if 0

#else // 解bank冲突版本

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

                    *(floatx4 *)(&out_smem_w8a16[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
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
                        floatx4 temp = *(floatx4 *)(&out_smem_w8a16[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
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

#endif
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
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_2_w8a16(
    const Element *input_ptr,
    const uint32_t *weight_ptr,
    Element *weight_scale_ptr,
    int max_n_len_offset,
    int size_n,
    union_vec_opt_w8a16_A<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt_w8a16<uint32_t, WARP_K / 16> B_int_reg[][STAGE],
    Element B_scale_reg[][STAGE],
    reg_bf16_fp16<typename ScalarType<Element>::scalar_t2> B_reg[][STAGE][WARP_K / 32],
    floatx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k (256)
    int seqlen_B_stride, // size_k (256)
    int top_k,
    const int *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx)
{

    const int size_k = seqlen_A_stride;
    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K_A = 32;
    constexpr int READ_K_B = 128;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    using scalar_t2 = typename ScalarType<Element>::scalar_t2;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int sorted_token_idx = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = ((std::min(sorted_token_idx / top_k, max_n_len_offset - 1)) * size_k + col_id * 32) * 2;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * WARP_N + n_tile * MFMA_N) * 128 + row_id * 32 + col_id * 32 * 16;
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K_A; k_tile++)
            {
                buffer_load_reg_dwordx4_half8(input_ptr, A_reg[m_tile][i].input_half8[k_tile], 0,
                                              g_row_A[m_tile] + k_tile * 8 * 2 + k_start * 2);
            }
        }
#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K_B; k_tile++)
            {
                int base_offset = g_row_B[n_tile] + k_tile * 128 * size_n + k_start_b;
                buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][i].int4_array[k_tile * 2], 0, base_offset);
                buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][i].int4_array[k_tile * 2 + 1], 0, base_offset + 16);
            }
        }
        k_start += stage_offset;
        k_start_b += stage_offset_b;
    }

    int k_loop_count = 0;
    const int max_k_loop = (size_k + stage_offset * STAGE - 1) / (stage_offset * STAGE);

#pragma unroll 1 // 允许编译器展开一次，但保留循环
    for (int loop = 0; loop < max_k_loop; ++loop)
    {
#pragma unroll
        for (int i = 0; i < STAGE; ++i)
        {
            int next_k_start = k_start + i * stage_offset;
            if (next_k_start < size_k)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K_A; k_tile++)
                    {
                        buffer_load_reg_dwordx4_half8(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].input_half8[k_tile], 0,
                                                      g_row_A[m_tile] + k_tile * 8 * 2 + (k_start + i * stage_offset) * 2);
                    }
                }
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K_B; k_tile++)
                    {
                        int base_offset = g_row_B[n_tile] + k_tile * 128 * size_n + (k_start_b + i * stage_offset_b);
                        buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile * 2], 0, base_offset);
                        buffer_load_reg_dwordx4_uint32(weight_ptr, B_int_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile * 2 + 1], 0, base_offset + 16);
                    }
                }
            }

            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K_A + WARP_N / MFMA_N * WARP_K / READ_K_B * 2) * (STAGE - 1));

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    dequant_w8a16_to_reg<scalar_t2>(
                        B_int_reg[n_tile][i].uint_array[j * 2],
                        B_int_reg[n_tile][i].uint_array[j * 2 + 1],
                        B_reg[n_tile][i][j]);
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(
                            vec4_Element<Element>{
                                A_reg[m_tile][i].input_half[k_tile * 8 + 0],
                                A_reg[m_tile][i].input_half[k_tile * 8 + 1],
                                A_reg[m_tile][i].input_half[k_tile * 8 + 2],
                                A_reg[m_tile][i].input_half[k_tile * 8 + 3]},
                            vec4_Element<Element>{
                                B_reg[n_tile][i][k_tile].vec2_i16[0][0],
                                B_reg[n_tile][i][k_tile].vec2_i16[0][1],
                                B_reg[n_tile][i][k_tile].vec2_i16[1][0],
                                B_reg[n_tile][i][k_tile].vec2_i16[1][1]},
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(
                            vec4_Element<Element>{
                                A_reg[m_tile][i].input_half[k_tile * 8 + 4],
                                A_reg[m_tile][i].input_half[k_tile * 8 + 5],
                                A_reg[m_tile][i].input_half[k_tile * 8 + 6],
                                A_reg[m_tile][i].input_half[k_tile * 8 + 7]},
                            vec4_Element<Element>{
                                B_reg[n_tile][i][k_tile].vec2_i16[2][0],
                                B_reg[n_tile][i][k_tile].vec2_i16[2][1],
                                B_reg[n_tile][i][k_tile].vec2_i16[3][0],
                                B_reg[n_tile][i][k_tile].vec2_i16[3][1]},
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        } // end of stage loop

        k_start += stage_offset * STAGE;
        k_start_b += stage_offset_b * STAGE;
    } // end of K loop

#pragma unroll
    for (int tail = 1; tail < STAGE; ++tail)
    {
        int stage_idx = tail; // 对应 A_reg/B_reg 中的索引（注意取模）
        vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K_A + WARP_N / MFMA_N * WARP_K / READ_K_B * 2) * (STAGE - 1 - tail));
#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                dequant_w8a16_to_reg<scalar_t2>(
                    B_int_reg[n_tile][stage_idx].uint_array[j * 2],
                    B_int_reg[n_tile][stage_idx].uint_array[j * 2 + 1],
                    B_reg[n_tile][stage_idx][j]);
            }
        }
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                {
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(
                        vec4_Element<Element>{
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 0],
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 1],
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 2],
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 3]},
                        vec4_Element<Element>{
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[0][0],
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[0][1],
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[1][0],
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[1][1]},
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element, float>(
                        vec4_Element<Element>{
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 4],
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 5],
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 6],
                            A_reg[m_tile][stage_idx].input_half[k_tile * 8 + 7]},
                        vec4_Element<Element>{
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[2][0],
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[2][1],
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[3][0],
                            B_reg[n_tile][stage_idx][k_tile].vec2_i16[3][1]},
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                }
            }
        }
    }

    extern __shared__ float out_smem_w8a16[];
    if constexpr (warp_k_num > 1)
    {
        constexpr int pading_n = WARP_N + 1;
        if (warp_k_id > 0)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
                    *(floatx4 *)(&out_smem_w8a16[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
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
                        floatx4 temp = *(floatx4 *)(&out_smem_w8a16[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
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
