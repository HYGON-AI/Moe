#ifndef MOE_W8A8_UTILS_HIP_H
#define MOE_W8A8_UTILS_HIP_H

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

#include "intrinsic_2.h"
#include "intrinsic.h"
#include "numeric_types.h"

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
__forceinline__ __device__ void gemm_nt_marlin_prefill(
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

    const int size_k = seqlen_A_stride;
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

    const int size_n = seqlen_B_stride;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        int token_ids = sorted_token_ids_element & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;

        g_row_A[m_tile] = (std::min(token_ids, max_n_len_offset - 1)) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * WARP_N + row_id + n_tile * MFMA_N) * 64 + col_id * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
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
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
            constexpr int epilogue_tile = (SIZE_K / BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 1; i < STAGE; ++i)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i - 1].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i - 1].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    extern __shared__ int out_smem[]; // 声明lds信息
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

#include "moe_w8a8_bk64_prefill.h"

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
__forceinline__ __device__ void gemm_nt_marlin_prefill_2(
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
    int32_t *sorted_token_ids_element_store,
    int *tok_ids_store,
    int *token_index_store,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    float tmp[4][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 4;

    const int size_k = seqlen_A_stride;

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
    const int size_n = seqlen_B_stride;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict
    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

    int A_index = warp_id;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + A_index * 4, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }
#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int index = (warp_n_id * (WARP_N / 32) + n_tile) * 2048 + (row_id & 7) * 32 + row_id / 8 * 1024 + col_id * 256;
        for (int i = 0; i < 2; i++)
        {
            g_row_B[n_tile * 2 + i] = index + i * 16;
        }
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
        int token_index = token_ids * real_topk /* top_k */ + topk_ids;
        g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * WARP_K + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
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
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
        }
    }

    for (kloop; kloop < SIZE_K / 128; kloop++)
    {

        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / MFMA_N));
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
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
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                {

#pragma unroll
                    for (int min_tile_m = 0; min_tile_m < WARP_M / 16; min_tile_m++)
                    {
#pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
                            int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                            tok_ids_store[it] = (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                            token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);

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
                vmcnt_only_wait(2 * (WARP_N / MFMA_N));

                i = 1;

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

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

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
                vmcnt_only_wait((WARP_N / MFMA_N) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));
            }
            else
            {
                vmcnt_only_wait((WARP_N / MFMA_N));
            }

            i = 0;

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

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    if (size_k % 128 == 64)
    {

        const int tail_kloop = size_k / 128;
        const int k_start_tail = warp_k_id * WARP_K + tail_kloop * 128;

        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 /* padding */ /* +(index*16)/ 4 */, (k_tile * READ_K + k_start_tail) / 4, (g_row_A[m_tile]) / 4);
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                vmcnt_wait((WARP_N / MFMA_N));

#pragma unroll
                for (int i = 0; i < 1; i++)
                {
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

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

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait(0);

            i = 0;
#pragma unroll
            for (int i = 0; i < 1; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
        else if constexpr (STAGE == 1)
        {
            ;
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
__forceinline__ __device__ void gemm_nt_marlin_decode_new(
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

    const int size_k = seqlen_A_stride;
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

    const int size_n = seqlen_B_stride;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        int token_ids = sorted_token_ids_element & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;

        g_row_A[m_tile] = (std::min(token_ids, max_n_len_offset - 1)) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * WARP_N + row_id + n_tile * MFMA_N) * 64 + col_id * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
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
            const int stage_prev = (i == 0) ? (STAGE - 1) : (i - 1);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][stage_prev].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][stage_prev].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
            constexpr int epilogue_tile = (SIZE_K / BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
                const int stage_prev = (i == 0) ? (STAGE - 1) : (i - 1);
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][stage_prev].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][stage_prev].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
                const int stage_prev = (i == 0) ? (STAGE - 1) : (i - 1);
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][stage_prev].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][stage_prev].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            int stage_idx = i + ii - 1;
            if (stage_idx >= STAGE)
                stage_idx -= STAGE;
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][stage_idx].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_idx].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 1; i < STAGE; ++i)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i - 1].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i - 1].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    extern __shared__ int out_smem[]; // 声明lds信息
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
__forceinline__ __device__ void gemm_nt_marlin_decode(
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

    const int size_k = seqlen_A_stride;
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

    const int size_n = seqlen_B_stride;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        int token_ids = sorted_token_ids_element & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;

        g_row_A[m_tile] = (std::min(token_ids, max_n_len_offset - 1)) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * WARP_N + n_tile * MFMA_N) * 64 +
                          row_id * 16 + col_id * 16 * 16;
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
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
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
            constexpr int epilogue_tile = (SIZE_K / BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 1; i < STAGE; ++i)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i - 1].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i - 1].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    extern __shared__ int out_smem[]; // 声明lds信息
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
          typename Element,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_2(
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
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 1;

    const int size_k = seqlen_A_stride;

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
    int kloop = 0;
    for (kloop; kloop < size_k / 128; kloop++)
    {

        const int size_n = seqlen_B_stride;
        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;
        const int stage_offset = warp_k_num * WARP_K;
        const int stage_offset_b = warp_k_num * WARP_K * size_n;
        const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

        int g_row_A[WARP_M / MFMA_M];
        int g_row_B[WARP_N / MFMA_N];

        auto g_input = tcp_cache_swizzle_func<64, Element>(input_ptr);
        auto g_sorted_token_ids_offset = tcp_cache_swizzle_func<64, int32_t>(sorted_token_ids_offset);
        int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1))];
            }
        }
        vmcnt_only_wait(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {

                int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
                int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
                int token_index = token_ids * real_topk /* top_k */ + topk_ids;
                g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
            g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
        }

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
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
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            i = 1;
            k_start_b += stage_offset_b;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                i = 1;
                k_start_b += stage_offset_b;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                vmcnt_wait(3 * (WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
                vmcnt_wait(2 * (WARP_N / MFMA_N));

                i = 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }

                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait((WARP_N / MFMA_N));

            i = 0;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
            vmcnt_wait(0);

            i = 1;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    if (size_k % 128 == 64)
    {
        const int size_n = seqlen_B_stride;
        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;
        const int stage_offset = warp_k_num * WARP_K;
        const int stage_offset_b = warp_k_num * WARP_K * size_n;
        const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

        int g_row_A[WARP_M / MFMA_M];
        int g_row_B[WARP_N / MFMA_N];

        auto g_input = tcp_cache_swizzle_func<64, Element>(input_ptr);
        auto g_sorted_token_ids_offset = tcp_cache_swizzle_func<64, int32_t>(sorted_token_ids_offset);
        int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1))];
            }
        }
        vmcnt_only_wait(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {

                int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
                int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
                int token_index = token_ids * real_topk /* top_k */ + topk_ids;
                g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
            g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
        }

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                vmcnt_wait((WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }

                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait(0);

            i = 0;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
        else if constexpr (STAGE == 1)
        {
            ;
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
          typename ElementAccum = int32_t,
          int SIZE_K,
          int N_LOOP_NUM = 1>
__forceinline__ __device__ void gemm_nt_marlin_prefill_n160_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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
    int32_t *sorted_token_ids_element_store,
    int *tok_ids_store,
    int *token_index_store,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    float tmp[4][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = N_LOOP_NUM;

    constexpr int size_k = SIZE_K;

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

    const int size_n = seqlen_B_stride;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
        {
            inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
        }
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int index = (warp_n_id * (WARP_N / 32) + n_tile) * 2048 + (row_id & 15) * 128 + col_id * 16;
        for (int i = 0; i < 2; i++)
        {
            g_row_B[n_tile * 2 + i] = index + i * 64;
        }
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
        {

            int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
            int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
            int token_index = token_ids /* top_k */;
            g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
        }
    }

    {

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < WARP_M / 16; min_tile_m++)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
                int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                tok_ids_store[it] = (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
            }
        }
    }

    int kloop = 0;
    for (kloop; kloop < size_k / 128; kloop++)
    {

        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
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
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / MFMA_N));
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
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
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);
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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(2 * (WARP_N / MFMA_N));

                i = 1;

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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

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

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            } // stage end

            n_loop--;
            vmcnt_only_wait((WARP_N / MFMA_N));

            i = 0;

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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            vmcnt_only_wait(0);

            i = 1;

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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

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

            __builtin_amdgcn_sched_barrier(0);
            vmcnt_wait(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    if (size_k % 128 == 64)
    {

        const int tail_kloop = size_k / 128;
        const int k_start_tail = warp_k_id * WARP_K + tail_kloop * 128;

        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start_tail) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                vmcnt_wait((WARP_N / MFMA_N));

#pragma unroll
                for (int i = 0; i < 2; i++)
                {
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

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

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait(0);

            i = 0;
#pragma unroll
            for (int i = 0; i < 1; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
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
        else if constexpr (STAGE == 1)
        {
            ;
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
          typename ElementAccum = int32_t,
          int SIZE_K,
          int N_LOOP_NUM = 4>
__forceinline__ __device__ void gemm_nt_marlin_prefill_2_n160_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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
    int32_t *sorted_token_ids_element_store,
    int *tok_ids_store,
    int *token_index_store,
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

    const int size_n = seqlen_B_stride;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

    int A_index = warp_id;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + A_index * 4, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int index = (warp_n_id * (WARP_N / 32) + n_tile) * 2048 + (row_id & 7) * 32 + row_id / 8 * 1024 + col_id * 256;
        for (int i = 0; i < 2; i++)
        {
            g_row_B[n_tile * 2 + i] = index + i * 16;
        }
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
        int token_index = token_ids * real_topk /* top_k */ + topk_ids;
        g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    {

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < WARP_M / 16; min_tile_m++)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
                int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                tok_ids_store[it] = (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
            }
        }
    }

    int kloop = 0;
    for (kloop; kloop < size_k / 128; kloop++)
    {

        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
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
                inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
            }
        }

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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / MFMA_N));
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
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
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);

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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(2 * (WARP_N / MFMA_N));

                i = 1;

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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
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
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            } // stage end

            n_loop--;
            vmcnt_only_wait((WARP_N / MFMA_N));

            i = 0;

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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            vmcnt_only_wait(0);

            i = 1;

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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            vmcnt_wait(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    if (size_k % 128 == 64)
    {

        const int tail_kloop = size_k / 128;
        const int k_start_tail = warp_k_id * WARP_K + tail_kloop * 128;

        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start_tail) / 4);
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                vmcnt_wait((WARP_N / MFMA_N));

#pragma unroll
                for (int i = 0; i < 2; i++)
                {
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait(0);

            i = 0;
#pragma unroll
            for (int i = 0; i < 1; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
        }
        else if constexpr (STAGE == 1)
        {
            ;
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
__forceinline__ __device__ void gemm_nt_marlin_decode_n160_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE], // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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

    const int size_n = seqlen_B_stride;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        int token_ids = sorted_token_ids_element & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;

        g_row_A[m_tile] = (std::min(token_ids, max_n_len_offset - 1)) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // align decode gemm1 with gemm2-shuffled weights
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
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
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
            constexpr int epilogue_tile = (SIZE_K / BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 1; i < STAGE; ++i)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i - 1].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i - 1].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    extern __shared__ float out_smem_float[]; // 声明lds信息
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

                    *(vec4_fp32 *)(&out_smem_float[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
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
                        vec4_fp32 temp = *(vec4_fp32 *)(&out_smem_float[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
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
          typename Element,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_2_n160_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 1;

    const int size_k = seqlen_A_stride;

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
    int kloop = 0;
    for (kloop; kloop < size_k / 128; kloop++)
    {

        const int size_n = seqlen_B_stride;
        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;
        const int stage_offset = warp_k_num * WARP_K;
        const int stage_offset_b = warp_k_num * WARP_K * size_n;
        const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

        int g_row_A[WARP_M / MFMA_M];
        int g_row_B[WARP_N / MFMA_N];

        auto g_input = tcp_cache_swizzle_func<64, Element>(input_ptr);
        auto g_sorted_token_ids_offset = tcp_cache_swizzle_func<64, int32_t>(sorted_token_ids_offset);
        int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1))];
            }
        }
        vmcnt_only_wait(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {

                int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
                int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
                int token_index = token_ids * real_topk /* top_k */ + topk_ids;
                g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
            g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
        }

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
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
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            i = 1;
            k_start_b += stage_offset_b;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                i = 1;
                k_start_b += stage_offset_b;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                vmcnt_wait(3 * (WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
                vmcnt_wait(2 * (WARP_N / MFMA_N));

                i = 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }

                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait((WARP_N / MFMA_N));

            i = 0;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
            vmcnt_wait(0);

            i = 1;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    if (size_k % 128 == 64)
    {
        const int size_n = seqlen_B_stride;
        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;
        const int stage_offset = warp_k_num * WARP_K;
        const int stage_offset_b = warp_k_num * WARP_K * size_n;
        const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

        int g_row_A[WARP_M / MFMA_M];
        int g_row_B[WARP_N / MFMA_N];

        auto g_input = tcp_cache_swizzle_func<64, Element>(input_ptr);
        auto g_sorted_token_ids_offset = tcp_cache_swizzle_func<64, int32_t>(sorted_token_ids_offset);
        int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1))];
            }
        }
        vmcnt_only_wait(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {

                int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
                int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
                int token_index = token_ids * real_topk /* top_k */ + topk_ids;
                g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
            g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
        }

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                vmcnt_wait((WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }

                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait(0);

            i = 0;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
        else if constexpr (STAGE == 1)
        {
            ;
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
__forceinline__ __device__ void gemm_nt_marlin_prefill_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE], // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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

    const int size_k = seqlen_A_stride;
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

    const int size_n = seqlen_B_stride;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        int token_ids = sorted_token_ids_element & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;

        g_row_A[m_tile] = (std::min(token_ids, max_n_len_offset - 1)) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * WARP_N + row_id + n_tile * MFMA_N) * 64 + col_id * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
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
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
            constexpr int epilogue_tile = (SIZE_K / BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 1; i < STAGE; ++i)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i - 1].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i - 1].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    extern __shared__ float out_smem_float[]; // 声明lds信息
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

                    *(vec4_fp32 *)(&out_smem_float[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
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
                        vec4_fp32 temp = *(vec4_fp32 *)(&out_smem_float[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
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
          typename Element,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_prefill_2_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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
    int32_t *sorted_token_ids_element_store,
    int *tok_ids_store,
    int *token_index_store,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    float tmp[4][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 4;

    const int size_k = seqlen_A_stride;

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

    const int size_n = seqlen_B_stride;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;
    const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
        {
            inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
        }
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int index = (warp_n_id * (WARP_N / 32) + n_tile) * 2048 + (row_id & 7) * 32 + row_id / 8 * 1024 + col_id * 256;
        for (int i = 0; i < 2; i++)
        {
            g_row_B[n_tile * 2 + i] = index + i * 16;
        }
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
        {

            int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
            int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
            int token_index = token_ids * real_topk /* top_k */ + topk_ids;
            g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
        }
    }

    int kloop = 0;
    for (kloop; kloop < size_k / 128; kloop++)
    {

        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
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
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
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
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / MFMA_N));
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
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
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                {

#pragma unroll
                    for (int min_tile_m = 0; min_tile_m < WARP_M / 16; min_tile_m++)
                    {
#pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
                            int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                            tok_ids_store[it] = (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                            token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);
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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(2 * (WARP_N / MFMA_N));

                i = 1;

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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

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

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            } // stage end

            n_loop--;
            vmcnt_only_wait((WARP_N / MFMA_N));

            i = 0;

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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            vmcnt_only_wait(0);

            i = 1;

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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

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

            __builtin_amdgcn_sched_barrier(0);
            vmcnt_wait(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    if (size_k % 128 == 64)
    {

        const int tail_kloop = size_k / 128;
        const int k_start_tail = warp_k_id * WARP_K + tail_kloop * 128;

        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start_tail) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile * 2 + it][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                vmcnt_wait((WARP_N / MFMA_N));

#pragma unroll
                for (int i = 0; i < 2; i++)
                {
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

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
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

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

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait(0);

            i = 0;
#pragma unroll
            for (int i = 0; i < 1; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
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
        else if constexpr (STAGE == 1)
        {
            ;
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
__forceinline__ __device__ void gemm_nt_marlin_decode_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE], // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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

    const int size_k = seqlen_A_stride;
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

    const int size_n = seqlen_B_stride;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        int token_ids = sorted_token_ids_element & 0x00FFFFFF;
        int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;

        g_row_A[m_tile] = (std::min(token_ids, max_n_len_offset - 1)) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
    {
        g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // align decode gemm1 with gemm2-shuffled weights
    }

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][i].int4_array[k_tile], 0, g_row_A[m_tile] + k_tile * 64 + k_start);
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
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
                    buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                }
            }
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
            constexpr int epilogue_tile = (SIZE_K / BLOCK_K - (STAGE - 1)) % STAGE; //( size_k/stage_offset - STAGE - 1) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
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
                        buffer_load_reg_dwordx4(input_ptr, A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start);
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(weight_ptr, B_reg[n_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile], 0, g_row_B[n_tile] + (i + 1) * stage_offset_b + k_tile * 64 * size_n + k_start_b);
                    }
                }
                vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1));

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][i].int8t_array[k_tile]),
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - ii));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 1; i < STAGE; ++i)
        {
            vmcnt_wait((WARP_M / MFMA_M * WARP_K / READ_K + WARP_N / MFMA_N * WARP_K / READ_K) * (STAGE - 1 - i));

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i - 1].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][i - 1].int8t_array[k_tile]),
                            C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
    }

    extern __shared__ float out_smem_float[]; // 声明lds信息
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

                    *(vec4_fp32 *)(&out_smem_float[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
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
                        vec4_fp32 temp = *(vec4_fp32 *)(&out_smem_float[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
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
          typename Element,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_2_fp8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],    // 2 = stage
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE], // 2 = stage
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
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
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 1;

    const int size_k = seqlen_A_stride;

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
    int kloop = 0;
    for (kloop; kloop < size_k / 128; kloop++)
    {

        const int size_n = seqlen_B_stride;
        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;
        const int stage_offset = warp_k_num * WARP_K;
        const int stage_offset_b = warp_k_num * WARP_K * size_n;
        const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

        int g_row_A[WARP_M / MFMA_M];
        int g_row_B[WARP_N / MFMA_N];

        auto g_input = tcp_cache_swizzle_func<64, Element>(input_ptr);
        auto g_sorted_token_ids_offset = tcp_cache_swizzle_func<64, int32_t>(sorted_token_ids_offset);
        int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1))];
            }
        }
        vmcnt_only_wait(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {

                int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
                int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
                int token_index = token_ids * real_topk /* top_k */ + topk_ids;
                g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
            g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
        }

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
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
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16 /* +2 */) * WARP_K / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            i = 1;
            k_start_b += stage_offset_b;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                i = 1;
                k_start_b += stage_offset_b;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                vmcnt_wait(3 * (WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }
                vmcnt_wait(2 * (WARP_N / MFMA_N));

                i = 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }

                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait((WARP_N / MFMA_N));

            i = 0;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
            vmcnt_wait(0);

            i = 1;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    if (size_k % 128 == 64)
    {
        const int size_n = seqlen_B_stride;
        int k_start = warp_k_id * WARP_K + kloop * (128);
        int k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;
        const int stage_offset = warp_k_num * WARP_K;
        const int stage_offset_b = warp_k_num * WARP_K * size_n;
        const int lds_stage_offset = WARP_M * WARP_K; // 默认GEMM2 WarpK = 64 , stage 最多为2 共享内存使用量为8K 不解bank conflict

        int g_row_A[WARP_M / MFMA_M];
        int g_row_B[WARP_N / MFMA_N];

        auto g_input = tcp_cache_swizzle_func<64, Element>(input_ptr);
        auto g_sorted_token_ids_offset = tcp_cache_swizzle_func<64, int32_t>(sorted_token_ids_offset);
        int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4, int(sorted_token_lens - 1))];
            }
        }
        vmcnt_only_wait(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {

                int token_ids = sorted_token_ids_element[m_tile] & 0x00FFFFFF;
                int topk_ids = (sorted_token_ids_element[m_tile] & 0xFF000000) >> 24;
                int token_index = token_ids * real_topk /* top_k */ + topk_ids;
                g_row_A[m_tile] = (std::min(token_index, max_n_len_offset - 1)) * size_k + row_id * 4;
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
        {
            g_row_B[n_tile] = (warp_n_id * (WARP_N / MFMA_N) + n_tile) * 16 * 64 + row_id * 16 + col_id * 16 * 16; // warp内部k连续[warpn, 64] warp外部在N方向连续
        }

        int i = 0;

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4 /* padding */ /* +(index*16)/ 4 */, 0, (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                }
            }

            k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 64;
                int i = 0;

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(cur_weight_ptr, B_reg[n_tile][stage_b_flag][i].int4_array[k_tile], 0, g_row_B[n_tile] + k_tile * 64 * size_n + k_start_b);
                    }
                }

                vmcnt_wait((WARP_N / MFMA_N));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                        }
                    }
                }

                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (128) * size_n;

            } // stage end

            n_loop--;
            vmcnt_wait(0);

            i = 0;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 + row_id * 64 + col_id * 16 /* padding */ /* + (row_id / 4) * 16 */ + (i)*WARP_M / 16 * (16 /* +2 */) * WARP_K]);
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
                        C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile] = mmac_fp8<Element>(
                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                            *(vec<Element, 8> *)(&B_reg[n_tile][stage_b_flag][i].int8t_array[k_tile]),
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile]);
                    }
                }
            }
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }
}

#endif
