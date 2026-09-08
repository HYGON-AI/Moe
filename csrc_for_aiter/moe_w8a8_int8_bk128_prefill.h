#pragma once

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
__forceinline__ __device__ void gemm_nt_marlin_prefill_up_int8_bk128(
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
            int token_index = token_ids;
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
    if (size_k / 128 > 0)
    {
        int k_start = warp_k_id * WARP_K;

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
    }

    for (kloop; kloop < size_k / 128; kloop++)
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

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;
                k_start_b = warp_k_id * WARP_K * size_n + kloop * (WARP_K * 2) * size_n;

            } // stage end

            n_loop--;
            if (kloop < (size_k / 128) - 1)
            {

                kloop++;
                int k_start = warp_k_id * WARP_K + kloop * (128);

                i = 0;

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
                kloop--;
            }

            vmcnt_only_wait((WARP_N / MFMA_N) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));

            if (kloop == (size_k / 128) - 1)
            {
                vmcnt_wait((WARP_N / MFMA_N));
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
            vmcnt_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));

            if (kloop == (size_k / 128) - 1)
            {
                vmcnt_wait(0);
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

            __builtin_amdgcn_sched_barrier(0);
            if (kloop == (size_k / 128) - 1)
            {
                vmcnt_wait(0);
            }
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
__forceinline__ __device__ void gemm_nt_marlin_prefill_down_int8_bk128(
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
        int token_index = token_ids * real_topk + topk_ids;
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
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
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac<Element>(
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
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    int N_LOOP_NUM,
    bool mul_topk_weight>
__attribute__((hcu_co_issue_vgpr_size(256)))
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_INT8_BK128(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk,
    bool tensorwise_scale)
{
    const int bidx = blockIdx.z;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.x;

    uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
    if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
        return;

    const uint32_t input_offset = bidz * BLOCK_SIZE_K;
    const int32_t delta_bidx = bidx;
    const int32_t expert_id = expert_ids[delta_bidx];
    constexpr int n_loop_num = N_LOOP_NUM;
    const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num;
    scalar_t *g_output = output + output_offset;

    if (expert_id == -1)
    {
        constexpr int N_thread = BLOCK_SIZE_N / 8;
        vec_element_8<scalar_t> zero_element_8;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            zero_element_8.data[i] = 0;
        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
        {
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
            const int32_t topk_ids_ep = (sorted_token_ids_element & 0xFF000000) >> 24;
            int token_index = token_ids * real_topk + topk_ids_ep;
            if (topk_ids_ep < real_topk)
            {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) = zero_element_8;
            }
        }
        return;
    }

    auto g_input = input;
    auto g_input_scale = input_scale;

    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 32;

    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    const int warp_n_num = BLOCK_SIZE_N / WARP_N;
    const int warp_k_num = BLOCK_SIZE_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    extern __shared__ Element smem[];
    Element *input_lds = (Element *)&(smem);
    Element *qweight_lds = input_lds;
    scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem));
    float *b_scale_lds = (float *)&(smem);

    union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
    union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

    if (size_k != 2048 && size_k != 3072 && size_k != 4096 && size_k != 6144 && size_k != 7168)
    {
        return;
    }

    const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;
    const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num;
    const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num;

    constexpr int store_size = WARP_M / 4;
    int token_index_store[store_size];
    int tok_ids_store[store_size];
    int32_t sorted_token_ids_element_store[store_size];
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
        for (int i = 0; i < 4; i++)
        {
            int m_idx = min_tile_m * 16 + i * 4 + col_id;
            int it = m_idx / 4;
            inline_buffer_load_dword(sorted_token_ids_element_store[it], m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
        }
    }

    auto g_qweight = qweight + qweight_offset;
    auto g_weight_scale = weight_scale + weight_scale_offset;
    intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
    __builtin_amdgcn_sched_barrier(0);
    if (size_k == 4096)
    {
        constexpr int SIZE_K = 4096;
        gemm_nt_marlin_prefill_up_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, nullptr, nullptr, real_topk);
    }
    else if (size_k == 6144)
    {
        constexpr int SIZE_K = 6144;
        gemm_nt_marlin_prefill_up_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, nullptr, nullptr, real_topk);
    }
    else if (size_k == 7168)
    {
        constexpr int SIZE_K = 7168;
        gemm_nt_marlin_prefill_up_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, nullptr, nullptr, real_topk);
    }
    else if (size_k == 3072)
    {
        constexpr int SIZE_K = 3072;
        gemm_nt_marlin_prefill_up_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, nullptr, nullptr, real_topk);
    }
    else if (size_k == 2048)
    {
        constexpr int SIZE_K = 2048;
        gemm_nt_marlin_prefill_up_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, nullptr, nullptr, real_topk);
    }
    __builtin_amdgcn_sched_barrier(0);

    float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

    float b_scale[n_loop_num][(WARP_N / mfma_n)];

    if (tensorwise_scale)
    {
        const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
        {
#pragma unroll
            for (int idx = 0; idx < WARP_N / mfma_n; idx++)
            {
                b_scale[n_loop][idx] = expert_b_scale;
            }
        }
    }
    else
    {
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
            {
                for (int i = 0; i < 32 / mfma_n; i++)
                {
                    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);
                    inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i], row_id * 2, b_scale_ptr_prepared, min_tile_n * 32 + i);
                }
            }
        }
    }

    float weight_dot_a_scale[WARP_M / mfma_m][4];
#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++)
    {
        for (int i = 0; i < 4; i++)
        {
            int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
            int token_ids = sorted_token_ids_element & 0x00FFFFFF;
            int token_index_safe = std::min(uint32_t(token_ids), size_m - 1);
            float input_scale_value = *(input_scale + token_index_safe * stride_asm);
            weight_dot_a_scale[idx][i] = input_scale_value;
        }
    }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
                const int index = row_id * 2 + min_tile_n * 32 + warp_n_id * WARP_N + BLOCK_SIZE_N * n_loop;
                const int tile_idx0 = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2;
                const int tile_idx1 = tile_idx0 + 1;
                const float b_scale0 = b_scale[n_loop][min_tile_n * 2];
                const float b_scale1 = b_scale[n_loop][min_tile_n * 2 + 1];
#pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++)
                {
                    const int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    if (tok_ids_store[it] < real_topk)
                    {
                        union
                        {
                            scalar_t scalar_array[2];
                            vec_element_2<scalar_t> vec_value;
                        } store_value;
                        const float scale = weight_dot_a_scale[min_tile_m][reg_id];
                        store_value.scalar_array[0] = b32_to_b16<scalar_t>(C_reg[n_loop][tile_idx0][reg_id] * scale * b_scale0);
                        store_value.scalar_array[1] = b32_to_b16<scalar_t>(C_reg[n_loop][tile_idx1][reg_id] * scale * b_scale1);
                        *reinterpret_cast<vec_element_2<scalar_t> *>(
                            &g_output[token_index_store[it] * size_n + index]) = store_value.vec_value;
                    }
                }
            }
        }
    }
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    int N_LOOP_NUM,
    bool mul_topk_weight> // true
__attribute__((hcu_co_issue_vgpr_size(256)))
__global__ void __launch_bounds__(512, 1) MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_INT8_BK128(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk,
    bool tensorwise_scale)
{
    const int bidx = blockIdx.z;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.x;

    uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
    if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
        return;

    const uint32_t input_offset = bidz * BLOCK_SIZE_K;
    const int32_t delta_bidx = bidx;
    const int32_t expert_id = expert_ids[delta_bidx];
    const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;

    auto g_input = input;
    auto g_input_scale = input_scale;

    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 32;

    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    const int warp_n_num = BLOCK_SIZE_N / WARP_N;
    const int warp_k_num = BLOCK_SIZE_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    extern __shared__ Element smem[];
    Element *input_lds = (Element *)&(smem);
    Element *qweight_lds = input_lds;
    scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem));
    float *b_scale_lds = (float *)&(smem);

    union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
    union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

    constexpr int n_loop_num = N_LOOP_NUM;

    if (size_k != 128 && size_k != 256 && size_k != 320 && size_k != 384 && size_k != 512 &&
        size_k != 640 && size_k != 768 && size_k != 1024 && size_k != 1280 && size_k != 2048)
    {
        return;
    }

    const uint64_t qweight_offset = expert_offset + bidy * 64 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
    const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
    const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
    scalar_t *g_output;
    g_output = output + output_offset;

    if (expert_id == -1)
    { // EP算法处理 epxert_id为-1 写回0
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
        vec_element_8<scalar_t> zero_element_8;

#pragma unroll
        for (int i = 0; i < 8; ++i)
            zero_element_8.data[i] = 0;

        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
        {
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
            const int32_t topk_ids_ep = (sorted_token_ids_element & 0xFF000000) >> 24;
            int token_index = token_ids * real_topk + topk_ids_ep;
            if (topk_ids_ep < real_topk)
            {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) = zero_element_8;
            }
        }
        return;
    }

    constexpr int store_size = WARP_M / 4;
    int token_index_store[store_size];
    int tok_ids_store[store_size];
    int32_t sorted_token_ids_element_store[store_size];
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
        for (int i = 0; i < 4; i++)
        {
            int m_idx = min_tile_m * 16 + i * 4 + col_id;
            int it = m_idx / 4;
            inline_buffer_load_dword(sorted_token_ids_element_store[it], m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
        }
    }

    auto g_qweight = qweight + qweight_offset;
    auto g_weight_scale = weight_scale + weight_scale_offset;
    intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
    if (size_k == 128)
    {
        constexpr int SIZE_K = 128;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 256)
    {
        constexpr int SIZE_K = 256;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 320)
    {
        constexpr int SIZE_K = 320;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 384)
    {
        constexpr int SIZE_K = 384;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 512)
    {
        constexpr int SIZE_K = 512;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 640)
    {
        constexpr int SIZE_K = 640;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 768)
    {
        constexpr int SIZE_K = 768;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 1024)
    {
        constexpr int SIZE_K = 1024;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 1280)
    {
        constexpr int SIZE_K = 1280;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }
    else if (size_k == 2048)
    {
        constexpr int SIZE_K = 2048;
        gemm_nt_marlin_prefill_down_int8_bk128<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element, scalar_t, SIZE_K, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, sorted_token_ids_element_store, tok_ids_store, token_index_store, real_topk);
    }

    float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

    float b_scale[n_loop_num][(WARP_N / mfma_n)];

    if (tensorwise_scale)
    {
        const float expert_b_scale = weight_scale[expert_id];
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
        {
#pragma unroll
            for (int idx = 0; idx < WARP_N / mfma_n; idx++)
            {
                b_scale[n_loop][idx] = expert_b_scale;
            }
        }
    }
    else
    {
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
        {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++)
            {
                for (int i = 0; i < 32 / mfma_n; i++)
                {

                    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

                    inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i], row_id * 2, b_scale_ptr_prepared, min_tile_n * 32 + i);
                }
            }
        }
    }

    float weight_dot_a_scale[WARP_M / mfma_m][4];
#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++)
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            const int it = idx * 4 + i;
            int token_index_safe = std::min(uint32_t(token_index_store[it]), size_m - 1);
            float input_scale_value = *(input_scale + token_index_safe * stride_asm);
            weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
        }
    }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
                const int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
                const int tile_idx0 = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2;
                const int tile_idx1 = tile_idx0 + 1;
                const float b_scale0 = b_scale[n_loop][min_tile_n * 2];
                const float b_scale1 = b_scale[n_loop][min_tile_n * 2 + 1];
#pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++)
                {
                    const int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    if (tok_ids_store[it] < real_topk)
                    {
                        union
                        {
                            scalar_t scalar_array[2];
                            vec_element_2<scalar_t> vec_value;
                        } store_value;
                        const float scale = weight_dot_a_scale[min_tile_m][reg_id];
                        store_value.scalar_array[0] = b32_to_b16<scalar_t>(C_reg[n_loop][tile_idx0][reg_id] * scale * b_scale0);
                        store_value.scalar_array[1] = b32_to_b16<scalar_t>(C_reg[n_loop][tile_idx1][reg_id] * scale * b_scale1);
                        *reinterpret_cast<vec_element_2<scalar_t> *>(
                            &g_output[token_index_store[it] * size_n + index]) = store_value.vec_value;
                    }
                }
            }
        }
    }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_up_int8_bk128_nloop(const GemmParams<T, T_hidden> &params)
{
    constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K); // std::cout<<"WARP_NUM\t"<<WARP_NUM;
    const bool mul_topk_weight = false;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    constexpr int n_loop_num = N_LOOP_NUM;
    static_assert(N_LOOP_NUM > 0, "N_LOOP_NUM must be positive");
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;

    const int lds_size = 20 * 1024;
    const int shared_mem_size = lds_size;
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向

    if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
        return;

    gridDim.y = DIVIDE(params.size_n, n_loop_num * BLOCK_SIZE_N); // n方向
    gridDim.x = 1;                                                // k方向
    MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_INT8_BK128<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                               GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_A_scale,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.stride_asm,
        params.stride_ask,
        params.stride_bse,
        params.stride_bsn,
        params.stride_bsk,
        params.sorted_token_lens,
        params.top_k,
        params.real_topk,
        params.tensorwise_scale);
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_up_int8_bk128(const GemmParams<T, T_hidden> &params)
{
    if (params.size_n == 640)
    {
        launch_moe_w8a8_prefill_up_int8_bk128_nloop<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, 1, T, T_hidden>(params);
    }
    else
    {
        launch_moe_w8a8_prefill_up_int8_bk128_nloop<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, 2, T, T_hidden>(params);
    }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, int N_LOOP_NUM, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_down_int8_bk128_nloop(const GemmParams<T, T_hidden> &params)
{
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    constexpr int n_loop_num = N_LOOP_NUM;
    static_assert(N_LOOP_NUM > 0, "N_LOOP_NUM must be positive");
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0)
        return;
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向    gridDim.x = 1; // k方向

    const int lds_size = BLOCK_SIZE_M * WARP_K * 2;
    const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
    const hipStream_t stream = at::cuda::getCurrentHIPStream();

    if (params.is_marlin == false)
    {
    }
    else
    {

        MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN_INT8_BK128<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                                     GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk,
            params.tensorwise_scale);
    }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w8a8_prefill_down_int8_bk128(const GemmParams<T, T_hidden> &params)
{
    launch_moe_w8a8_prefill_down_int8_bk128_nloop<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, 4, T, T_hidden>(params);
}
