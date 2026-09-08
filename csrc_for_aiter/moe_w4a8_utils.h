#ifndef MOE_W4A8_UTILS_HIP_H
#define MOE_W4A8_UTILS_HIP_H

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

#include "intrinsic_2.h"
#include "intrinsic.h"
#include "numeric_types.h"

template <
    int WARP_NUM,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int STAGE,
    int SIZE_K,
    typename Element>
__forceinline__ __device__ void w4a8_mmac_tail64(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    intx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int size_n,
    int size_k,
    const int *g_row_A,
    const int *g_row_B,
    int n_loop_num)
{
    if constexpr (WARP_K == 64)
    {
        if (size_k % 128 == 64)
        {
            __syncthreads();

            constexpr int MFMA_M = 16;
            constexpr int MFMA_N = 16;
            constexpr int MFMA_K = 32;
            constexpr int READ_K = 64;
            constexpr int warp_k_num = BLOCK_K / WARP_K;

            const int lane_id = threadIdx.x & 63;
            const int row_id = lane_id % 16;
            const int col_id = lane_id / 16;
            const int warp_k_id = warp_id % warp_k_num;
            const int warp_n_id = warp_id / warp_k_num;
            const int A_index = warp_id;
            const int k_start = warp_k_id * WARP_K + (size_k / 128) * 128;
            const int k_start_b = warp_k_id * (WARP_K / 2) * size_n + (size_k / 128) * (128 / 2) * size_n;
            auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    inline_buffer_load_dword_lds(
                        A_lds,
                        g_input,
                        (m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                        (k_tile * READ_K + k_start) / 4,
                        g_row_A[m_tile] / 4);
                }
            }

            vmcnt_only_wait(0);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    int s_index = m_tile * 16 * WARP_K;
                    A_reg[m_tile][0].int4_array[k_tile] =
                        *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
                }
            }

#pragma unroll
            for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
            {
                const Element *cur_weight_ptr = weight_ptr + n_loop * BLOCK_N * 32;
                union
                {
                    vec<int, 4> v;
                    uint32_t i[4];
                } weight_reg_tmp[WARP_N / 32];

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4_w4a8(
                            cur_weight_ptr,
                            weight_reg_tmp[n_tile].v,
                            k_tile * 32 * size_n + k_start_b,
                            g_row_B[n_tile]);
                    }
                }

                vmcnt_only_wait(0);
                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] =
                                    mmac<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
            }
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
__forceinline__ __device__ void gemm_nt_marlin_prefill_w4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    intx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 1
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    float tmp[4][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 4;

    const int size_k = SIZE_K == 0 ? seqlen_A_stride : SIZE_K;

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        int index = s_index + row_id * 2 * 8 + col_id * 32 * 8;

        g_row_B[n_tile] = index;
    }

    int A_index = warp_id;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(s_offset + col_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile] / top_k, max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < size_k / 128; kloop++)
    {
        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            i = 1;
            k_start_b += stage_offset_b;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_only_wait(0);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16 /* +2 */) * 64 + +(i)*WARP_M / 16 * (16 /* +2 */) * WARP_K;
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */]);
                    }
                }
            }
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                i = 1;
                k_start_b += stage_offset_b;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            } // stage end

            n_loop--;
            if (kloop < (size_k / 128) - 1)
            {

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));

            if (kloop == (size_k / 128) - 1)
            {
                vmcnt_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));

            if (kloop == (size_k / 128) - 1)
            {
                vmcnt_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    w4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, size_k, g_row_A, g_row_B, n_loop_num);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                    }
                }
            }
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
          typename ElementAccum = int32_t,
          int N_LOOP_NUM = 4>
__forceinline__ __device__ void gemm_nt_marlin_prefill_2_w4a8_k192(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    intx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k,
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[N_LOOP_NUM][(WARP_N / 16)],
    float tmp[N_LOOP_NUM][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk)
{

    constexpr int n_loop_num = N_LOOP_NUM;
    static_assert(SIZE_K == 192, "gemm_nt_marlin_prefill_2_w4a8_k192 only supports K=192");

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    const int size_n = scale_B_stride_e;
    int A_index = warp_id;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(s_offset + col_id, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        g_row_B[n_tile] = s_index + row_id * 2 * 8 + col_id * 32 * 8;
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile], max_n_len_offset - 1)) * SIZE_K + row_id * 4;
    }

#pragma unroll
    for (int k_block = 0; k_block < SIZE_K / 64; k_block++)
    {
        const int k_start = warp_k_id * WARP_K + k_block * 64;
        const int k_start_b = warp_k_id * (WARP_K / 2) * size_n + k_block * 32 * size_n;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                inline_buffer_load_dword_lds(A_lds, g_input,
                                             (m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                                             (k_tile * READ_K + k_start) / 4,
                                             g_row_A[m_tile] / 4);
            }
        }

        vmcnt_only_wait(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                int s_index = m_tile * 16 * WARP_K;
                A_reg[m_tile][0].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
            }
        }

#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
        {
            const Element *cur_weight_ptr = weight_ptr + n_loop * BLOCK_N * 32;
            union
            {
                vec<int, 4> v;
                uint32_t i[4];
            } weight_reg_tmp[WARP_N / 32];

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }

            vmcnt_only_wait(0);
            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                            C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                    }
                }
            }
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
          int N_LOOP_NUM,
          typename Element,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_prefill_w4a8_gemm1n256(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    intx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 1
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[N_LOOP_NUM][(WARP_N / 16)],
    float tmp[N_LOOP_NUM][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = N_LOOP_NUM;

    const int size_k = SIZE_K == 0 ? seqlen_A_stride : SIZE_K;

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        int index = s_index + row_id * 2 * 8 + col_id * 32 * 8;

        g_row_B[n_tile] = index;
    }

    int A_index = warp_id;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(s_offset + col_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile] / top_k, max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < size_k / 128; kloop++)
    {
        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            i = 1;
            k_start_b += stage_offset_b;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / 32));

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16 /* +2 */) * 64 + +(i)*WARP_M / 16 * (16 /* +2 */) * WARP_K;
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */]);
                    }
                }
            }

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                i = 1;
                k_start_b += stage_offset_b;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            } // stage end

            n_loop--;
            if (kloop < (size_k / 128) - 1)
            {

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));

            if (kloop == (size_k / 128) - 1)
            {
                vmcnt_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));

            if (kloop == (size_k / 128) - 1)
            {
                vmcnt_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    w4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, size_k, g_row_A, g_row_B, n_loop_num);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                    }
                }
            }
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
          typename ElementAccum = int32_t,
          int N_LOOP_NUM = 4>
__forceinline__ __device__ void gemm_nt_marlin_prefill_2_w4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    intx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 1
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[N_LOOP_NUM][(WARP_N / 16)],
    float tmp[N_LOOP_NUM][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = N_LOOP_NUM;

    const int size_k = SIZE_K;

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int A_index = warp_id;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(s_offset + col_id, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        int index = s_index + row_id * 2 * 8 + col_id * 32 * 8;

        g_row_B[n_tile] = index;
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile], max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < SIZE_K / 128; kloop++)
    {

        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            i = 1;
            k_start_b += stage_offset_b;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / 32));

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16 /* +2 */) * 64 + +(i)*WARP_M / 16 * (16 /* +2 */) * WARP_K;
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */]);
                    }
                }
            }

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                i = 1;
                k_start_b += stage_offset_b;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            } // stage end

            n_loop--;

            if (kloop < (SIZE_K / 128) - 1)
            {
                __syncthreads();

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));
            }
            else
            {
                vmcnt_only_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));
            }
            else
            {
                vmcnt_only_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    w4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, size_k, g_row_A, g_row_B, n_loop_num);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                    }
                }
            }
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
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_w4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE], // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][STAGE], // 2 = stage
    intx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 8
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx)
{

    int size_k = seqlen_A_stride;
    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K / 2 * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K / 2 * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];

        g_row_A[m_tile] = (std::min(sorted_token_ids_element / top_k, max_n_len_offset - 1)) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        int index = s_index + row_id * 2 * 8 + col_id * 32 * 8;
        g_row_B[n_tile] = index;
    }

    vec<char, 8> vec8_ones = (vec<char, 8>){1, 1, 1, 1, 1, 1, 1, 1};
    vec<float, 4> vec4_ones = {0, 0, 0, 0};

    union
    {
        vec<int, 4> v;
        uint32_t i[4];
    } weight_reg_tmp[WARP_N / 32][WARP_K / READ_K][STAGE];

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                A_reg[m_tile][i].int4_array[k_tile] = *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + k_tile * 64 + k_start]);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4_w4a8(weight_ptr, weight_reg_tmp[n_tile][k_tile][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
            }
        }
        k_start += stage_offset;
        k_start_b += stage_offset_b;
    }
    k_start -= stage_offset;
    k_start_b -= stage_offset_b;

    for (; k_start < size_k - stage_offset * STAGE; k_start += stage_offset * STAGE, k_start_b += stage_offset_b * STAGE)
    {

#pragma unroll
        for (int i = 0; i < STAGE; ++i)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile] = *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start]);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v = *(vec<uint, 4> *)(&weight_ptr[k_tile * 32 * size_n + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                }
            }
            vmcnt_wait(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                    for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                    {
#pragma unroll
                        for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                        {
                            packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][i].i[it * 2 + k_tile_second]));
                        }
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {

                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
        } // stage end

    } // k_loop end

    if (k_start + stage_offset < size_k)
    { // size_k / block 为偶数
        int i = 0;
        if constexpr (STAGE == 4)
        {
            int epilogue_tile = (size_k / BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile] = *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start]);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v = *(vec<uint, 4> *)(&weight_ptr[k_tile * 32 * size_n + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                    }
                }
                vmcnt_wait(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                        for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                        {
#pragma unroll
                            for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                            {
                                packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][i].i[it * 2 + k_tile_second]));
                            }
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {

                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
            }
        }
        else
        { // two stage
            constexpr int epilogue_tile = 1;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile] = *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start]);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v = *(vec<int, 4> *)(&weight_ptr[k_tile * 32 * size_n + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                    }
                }
                vmcnt_wait(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                        for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                        {
#pragma unroll
                            for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                            {
                                packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][i].i[it * 2 + k_tile_second]));
                            }
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {

                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                    C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            vmcnt_wait(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                    for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                    {
#pragma unroll
                        for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                        {
                            packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][(i + ii - 1) % STAGE].i[it * 2 + k_tile_second]));
                        }
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {

                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
        }
    }
    else
    {
    }

    extern __shared__ int out_smem[];
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

                    *(intx4 *)(&out_smem[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
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
                        intx4 temp = *(intx4 *)(&out_smem[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
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
__forceinline__ __device__ void gemm_nt_marlin_decode_2_w4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    intx4 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride, // size_k: 7168
    int seqlen_B_stride, // size_k: 7168
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k, // 1
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    float tmp[4][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 4;

    const int size_k = SIZE_K;

    int lane_id = threadIdx.x & 63; // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64; // k方向 4个线程 每个线程读16个int8
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int A_index = warp_id;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(s_offset + col_id, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        int index = s_index + row_id * 2 * 8 + col_id * 32 * 8;

        g_row_B[n_tile] = index;
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile], max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < SIZE_K / 128; kloop++)
    {

        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            i = 1;
            k_start_b += stage_offset_b;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / 32));

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16 /* +2 */) * 64 + +(i)*WARP_M / 16 * (16 /* +2 */) * WARP_K;
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */]);
                    }
                }
            }

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                i = 1;
                k_start_b += stage_offset_b;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                            uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            } // stage end

            n_loop--;

            if (kloop < (SIZE_K / 128) - 1)
            {

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));
            }
            else
            {
                vmcnt_only_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));
            }
            else
            {
                vmcnt_only_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = packed_val[k_tile] & 0xf0f0f0f0;
                        uint32_t packed_val_second = (packed_val[k_tile] << 4) & 0xf0f0f0f0;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    w4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, size_k, g_row_A, g_row_B, n_loop_num);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                    }
                }
            }
        }
    }
}

#endif
