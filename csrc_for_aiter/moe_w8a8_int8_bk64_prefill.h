#pragma once

#ifndef MOE_W8A8_FLAT_ROUTE_HELPERS
#define MOE_W8A8_FLAT_ROUTE_HELPERS
__forceinline__ __device__ void moe_w8a8_setprio_high()
{
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio 1");
}

__forceinline__ __device__ void moe_w8a8_setprio_normal()
{
    asm volatile("s_setprio 0");
    __builtin_amdgcn_sched_barrier(0);
}

__forceinline__ __device__ int moe_w8a8_route_token_id(
    const int32_t route_id,
    const uint32_t real_topk)
{
    return route_id / real_topk;
}

__forceinline__ __device__ int moe_w8a8_route_topk_id(
    const int32_t route_id,
    const uint32_t real_topk)
{
    const int token_id = moe_w8a8_route_token_id(route_id, real_topk);
    return route_id - token_id * real_topk;
}
#endif

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
          typename scalar_t,
          int SIZE_K,
          int N_LOOP_NUM = 1>
__forceinline__ __device__ void gemm_nt_marlin_prefill_n160_int8_bk64(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][1],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][1],
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
    int32_t *sorted_token_ids_element_store,
    int *tok_ids_store,
    int *token_index_store,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    uint32_t real_topk)
{

    constexpr int n_loop_num = N_LOOP_NUM;
    constexpr int size_k = SIZE_K;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    const int size_n = seqlen_B_stride;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

    if constexpr (WARP_NUM >= MFMA_M / 4)
    {
        if constexpr (WARP_NUM == MFMA_M / 4)
        {
            const int A_index = warp_id;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
                inline_buffer_load_dword(
                    sorted_token_ids_element[m_tile],
                    std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + A_index * 4,
                             int(sorted_token_lens - 1)),
                    g_sorted_token_ids_offset,
                    0);
            }
        }
        else if (warp_id < MFMA_M / 4)
        {
            const int A_index = warp_id;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
                inline_buffer_load_dword(
                    sorted_token_ids_element[m_tile],
                    std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + A_index * 4,
                             int(sorted_token_lens - 1)),
                    g_sorted_token_ids_offset,
                    0);
            }
        }
    }
    else
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                inline_buffer_load_dword(
                    sorted_token_ids_element[m_tile],
                    std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + index * 4,
                             int(sorted_token_lens - 1)),
                    g_sorted_token_ids_offset,
                    0);
            }
        }
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int index = (warp_n_id * (WARP_N / 32) + n_tile) * 2048 +
                    (row_id) * 16 /* + row_id / 8 * 1024 */ + col_id * 16 * 16;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            g_row_B[n_tile * 2 + i] = index + i * 16 * 64;
        }
    }

    vmcnt_only_wait(0);

    if constexpr (WARP_NUM >= MFMA_M / 4)
    {
        if constexpr (WARP_NUM == MFMA_M / 4)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
                int token_index = moe_w8a8_route_token_id(
                    sorted_token_ids_element[m_tile], real_topk);
                g_row_A[m_tile] = std::min(token_index, max_n_len_offset - 1) * size_k + row_id * 4;
            }
        }
        else if (warp_id < MFMA_M / 4)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
                int token_index = moe_w8a8_route_token_id(
                    sorted_token_ids_element[m_tile], real_topk);
                g_row_A[m_tile] = std::min(token_index, max_n_len_offset - 1) * size_k + row_id * 4;
            }
        }
    }
    else
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
            for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
            {
                int token_index = moe_w8a8_route_token_id(
                    sorted_token_ids_element[m_tile], real_topk);
                g_row_A[m_tile] = std::min(token_index, max_n_len_offset - 1) * size_k + row_id * 4;
            }
        }
    }

    const int A_index = warp_id;

    if (sorted_token_ids_element_store != nullptr)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < WARP_M / 16; min_tile_m++)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
                const int32_t route_id = sorted_token_ids_element_store[it];
                tok_ids_store[it] = moe_w8a8_route_topk_id(route_id, real_topk);
                token_index_store[it] = route_id;
            }
        }
    }

    constexpr int k64_loop_num = size_k / READ_K;
    for (int k64_loop = 0; k64_loop < k64_loop_num; k64_loop++)
    {
        const int k_start = warp_k_id * WARP_K + k64_loop * stage_offset;
        const int k_start_b = warp_k_id * WARP_K * size_n + k64_loop * stage_offset_b;

        if (k64_loop == 0)
        {
            if constexpr (WARP_NUM == MFMA_M / 4)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(
                            A_lds,
                            g_input,
                            (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4,
                            0,
                            (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                    }
                }
            }
            else if constexpr (WARP_NUM > MFMA_M / 4)
            {
                if (warp_id < MFMA_M / 4)
                {
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            inline_buffer_load_dword_lds(
                                A_lds,
                                g_input,
                                (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4,
                                0,
                                (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                        {
                            inline_buffer_load_dword_lds(
                                A_lds,
                                g_input,
                                (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4,
                                0,
                                (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                        }
                    }
                }
            }
        }

        const Element *cur_weight_ptr = weight_ptr;
        int stage_b_flag = 0;

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
        {
#pragma unroll
            for (int it = 0; it < 2; it++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(
                        cur_weight_ptr,
                        B_reg[n_tile * 2 + it][stage_b_flag][0].int4_array[k_tile],
                        0,
                        g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                }
            }
        }

        vmcnt_wait((WARP_N / 16) * (WARP_K / READ_K));

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                A_reg[m_tile][0].int4_array[k_tile] =
                    *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 +
                                                 row_id * 64 + col_id * 16]);
            }
        }

        for (int n_loop = 1; n_loop < n_loop_num; n_loop++)
        {
            int next_stage_b_flag = stage_b_flag ^ 1;
            cur_weight_ptr += BLOCK_N * 64;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(
                            cur_weight_ptr,
                            B_reg[n_tile * 2 + it][next_stage_b_flag][0].int4_array[k_tile],
                            0,
                            g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }

            vmcnt_wait(WARP_N / MFMA_N);

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
                            C_reg[n_loop - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] =
                                mmac<Element>(
                                    *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][0].int8t_array[k_tile]),
                                    C_reg[n_loop - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            stage_b_flag = next_stage_b_flag;
        }

        vmcnt_wait(0);

        if (k64_loop + 1 < k64_loop_num)
        {
            S_BARRIER;
            const int next_k_start = warp_k_id * WARP_K + (k64_loop + 1) * stage_offset;
            if constexpr (WARP_NUM == MFMA_M / 4)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(
                            A_lds,
                            g_input,
                            (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4,
                            0,
                            (g_row_A[m_tile] + k_tile * READ_K + next_k_start) / 4);
                    }
                }
            }
            else if constexpr (WARP_NUM > MFMA_M / 4)
            {
                if (warp_id < MFMA_M / 4)
                {
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            inline_buffer_load_dword_lds(
                                A_lds,
                                g_input,
                                (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4,
                                0,
                                (g_row_A[m_tile] + k_tile * READ_K + next_k_start) / 4);
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        for (int index = warp_id; index < MFMA_M / 4; index += WARP_NUM)
                        {
                            inline_buffer_load_dword_lds(
                                A_lds,
                                g_input,
                                (m_tile * (16 /* +2 */) * WARP_K + index * 4 * WARP_K) / 4,
                                0,
                                (g_row_A[m_tile] + k_tile * READ_K + next_k_start) / 4);
                        }
                    }
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
                        C_reg[n_loop_num - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] =
                            mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][0].int8t_array[k_tile]),
                                C_reg[n_loop_num - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
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
          typename ElementAccum = int32_t,
          int SIZE_K,
          int N_LOOP_NUM = 4>
__forceinline__ __device__ void gemm_nt_marlin_prefill_2_n160_int8_bk64(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][1],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][1],
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
    int32_t *sorted_token_ids_element_store,
    int *tok_ids_store,
    int *token_index_store,
    uint32_t real_topk)
{

    constexpr int n_loop_num = N_LOOP_NUM;
    constexpr int size_k = SIZE_K;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    const int size_n = seqlen_B_stride;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

    int A_index = warp_id;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        inline_buffer_load_dword(
            sorted_token_ids_element[m_tile],
            std::min(bidx * BLOCK_M + m_tile * MFMA_M + col_id + A_index * 4,
                     int(sorted_token_lens - 1)),
            g_sorted_token_ids_offset,
            0);
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int index = (warp_n_id * (WARP_N / 32) + n_tile) * 2048 +
                    (row_id & 7) * 32 + row_id / 8 * 1024 + col_id * 256;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            g_row_B[n_tile * 2 + i] = index + i * 16;
        }
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int token_index = sorted_token_ids_element[m_tile];
        g_row_A[m_tile] = std::min(token_index, max_n_len_offset - 1) * size_k + row_id * 4;
    }

    if (sorted_token_ids_element_store != nullptr)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < WARP_M / 16; min_tile_m++)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
                const int32_t route_id = sorted_token_ids_element_store[it];
                tok_ids_store[it] = moe_w8a8_route_topk_id(route_id, real_topk);
                token_index_store[it] = route_id;
            }
        }
    }

    constexpr int k64_loop_num = size_k / READ_K;
    for (int k64_loop = 0; k64_loop < k64_loop_num; k64_loop++)
    {
        const int k_start = warp_k_id * WARP_K + k64_loop * stage_offset;
        const int k_start_b = warp_k_id * WARP_K * size_n + k64_loop * stage_offset_b;

        if (k64_loop == 0)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    inline_buffer_load_dword_lds(
                        A_lds,
                        g_input,
                        (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4,
                        0,
                        (g_row_A[m_tile] + k_tile * READ_K + k_start) / 4);
                }
            }
        }

        const Element *cur_weight_ptr = weight_ptr;
        int stage_b_flag = 0;

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
        {
#pragma unroll
            for (int it = 0; it < 2; it++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    buffer_load_reg_dwordx4(
                        cur_weight_ptr,
                        B_reg[n_tile * 2 + it][stage_b_flag][0].int4_array[k_tile],
                        0,
                        g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                }
            }
        }

        vmcnt_wait((WARP_N / 16) * (WARP_K / READ_K));

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                A_reg[m_tile][0].int4_array[k_tile] =
                    *(vec<Element, 16> *)(&A_lds[m_tile * (16 /* +2 */) * 64 +
                                                 row_id * 64 + col_id * 16]);
            }
        }

        for (int n_loop = 1; n_loop < n_loop_num; n_loop++)
        {
            int next_stage_b_flag = stage_b_flag ^ 1;
            cur_weight_ptr += BLOCK_N * 64;

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        buffer_load_reg_dwordx4(
                            cur_weight_ptr,
                            B_reg[n_tile * 2 + it][next_stage_b_flag][0].int4_array[k_tile],
                            0,
                            g_row_B[n_tile * 2 + it] + k_tile * 64 * size_n + k_start_b);
                    }
                }
            }

            vmcnt_wait(WARP_N / MFMA_N);

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
                            C_reg[n_loop - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] =
                                mmac<Element>(
                                    *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][0].int8t_array[k_tile]),
                                    C_reg[n_loop - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            stage_b_flag = next_stage_b_flag;
        }

        vmcnt_wait(0);

        if (k64_loop + 1 < k64_loop_num)
        {
            S_BARRIER;
            const int next_k_start = warp_k_id * WARP_K + (k64_loop + 1) * stage_offset;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    inline_buffer_load_dword_lds(
                        A_lds,
                        g_input,
                        (m_tile * (16 /* +2 */) * WARP_K + A_index * 4 * WARP_K) / 4,
                        0,
                        (g_row_A[m_tile] + k_tile * READ_K + next_k_start) / 4);
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
                        C_reg[n_loop_num - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] =
                            mmac<Element>(
                                *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][0].int8t_array[k_tile]),
                                C_reg[n_loop_num - 1][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                    }
                }
            }
        }
    }
}
