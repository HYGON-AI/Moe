
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "moe_wna16_utils.h"

#include "wait.h"
#include "intrinsic.h"
#include <hip/hip_fp16.h> // 确保包含头文件
#include <cuda_fp16.h>

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

template <typename scalar_t,
          int bit,
          uint32_t top_k,
          uint32_t BLOCK_SIZE_M,
          uint32_t BLOCK_SIZE_N,
          uint32_t BLOCK_SIZE_K,
          bool has_zp,
          bool mul_topk_weight,
          uint32_t group_size_n,
          uint32_t group_size_k,
          int BLOCK_SIZE_M_LOOPS,
          int BLOCK_SIZE_N_LOOPS,
          int BLOCK_SIZE_K_LOOPS,
          bool USE_ATOMIC,
          int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(256, 1) moe_w8a8_gemm_kernel_block_wise_kernel2(

    const uint32_t *__restrict__ input,
    const float *__restrict__ input_scales,
    float *__restrict__ output,
    const uint32_t *__restrict__ qweight,
    int *b_qweight_out,
    const float *__restrict__ scales,
    const uint32_t *__restrict__ qzeros,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    int num_token_blocks,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k)
{

  int block_kid = blockIdx.x;
  int block_nid = blockIdx.y;
  int block_mid = blockIdx.z;

  if (block_mid * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;
  constexpr int WARP_N = 16;
  constexpr int WARP_NUM = BLOCK_SIZE_N / WARP_N;
  const uint32_t GROUPS = BLOCK_SIZE_K / group_size_k;

  auto g_input = tcp_cache_swizzle_func<64, uint32_t>(input);
  auto g_sorted_token_ids = tcp_cache_swizzle_func<1, int32_t>(sorted_token_ids);

  int warp_id_vec = (threadIdx.x / 64); // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = (threadIdx.x & 63); // lane id, 0-63

  int laneid_shfl_4 = lane_id >> 4;
  int laneid_and_15 = lane_id & 15;
  int w_lane_n_idx = laneid_shfl_4; // ((WARP_N/16 == 2) ?((laneid_shfl_4 & 1)*2 + (laneid_shfl_4 >> 1)) : laneid_shfl_4); // (0, 1, 2, 3) --> (0, 2, 1, 3) //laneid_shfl_4

  int w_lane_k_idx = laneid_and_15;
  int group_size_m = 128;

  int k_loop_num = DIVIDE(size_k, gridDim.x * BLOCK_SIZE_K);

  using Dtype = ScalarType<scalar_t>;

  using scalar_t2 = typename ScalarType<Float16>::scalar_t2;

  int expert_id = expert_ids[block_mid];
  if (expert_ids[block_mid] == -1)
    return;
  extern __shared__ uint8_t block_input_tmp_w8a8[];
  uint32_t *weight_lds = reinterpret_cast<uint32_t *>(block_input_tmp_w8a8); // lds
  int *block_input_vec4_int8 = reinterpret_cast<int *>(block_input_tmp_w8a8);
  union_vec_array<BLOCK_SIZE_M> token_index_arr;
  int token_index_arr_new[(BLOCK_SIZE_M + 15) / 16];
  int token_index_arr_32[(BLOCK_SIZE_M + 15) / 16][((BLOCK_SIZE_M + WARP_NUM - 1) / WARP_NUM) / 2];

  constexpr int K_TILE = 64;
  constexpr int32_t pack_factor = 32 / bit; // 4字节能存多少元素
  constexpr int STAGES = 2;
  constexpr int nums_token_a_warp_load = (16 + WARP_NUM - 1) / WARP_NUM;

  scalar_t2 scale_f2[2];
  scalar_t2 qzero_f2[2];

  const uint32_t expert_offset_weight = ((uint32_t)size_n) * size_k * expert_id;

  const uint32_t *expert_qweight = qweight + expert_offset_weight / pack_factor;
  auto g_expert_qweight = tcp_cache_swizzle_func<1, uint32_t>(expert_qweight);

  const float *expert_scales = scales + expert_offset_weight / (group_size_k * group_size_n);
  const float *expert_scales_input = input_scales;

  union_vec8_int8 expert_w_reg[BLOCK_SIZE_K / K_TILE][((WARP_N + 31) / 32) * (K_TILE / 64) * 2][2];

  union_vec<float, GROUPS> expert_scales_groups[(WARP_N + 31) / 32][WARP_N / 16];
  float input_scales_float[(BLOCK_SIZE_M + 15) / 16];
  float final_scales[(BLOCK_SIZE_M + 15) / 16];

  union_vec8_int8 input_vec8_int8[(BLOCK_SIZE_M + 15) / 16][(BLOCK_SIZE_K / 32)];
  int n_loop_num;

  if (BLOCK_SIZE_N_LOOPS == 1)
  {
    n_loop_num = 1;
  }
  else if (BLOCK_SIZE_N_LOOPS == 2)
  {
    n_loop_num = 2;
  }
  else if (BLOCK_SIZE_N_LOOPS == 4)
  {
    n_loop_num = 4;
  }
  else
  {
    static_assert("NOT SUPPORT");
  }

  vec4_int32 res_tmp[(BLOCK_SIZE_M + 15) / 16][((WARP_N + 31) / 32) * (32 / 16)] = {0};
  vec4_fp32 res[(BLOCK_SIZE_M + 15) / 16][((WARP_N + 31) / 32) * (32 / 16)][BLOCK_SIZE_N_LOOPS] = {0};
  vec4_fp32 res_float[(BLOCK_SIZE_M + 15) / 16][((WARP_N + 31) / 32) * (32 / 16)][BLOCK_SIZE_N_LOOPS] = {0};
  vec4_int32 res_zeros = {0};

  for (int m = 0; m < (BLOCK_SIZE_M + 15) / 16; m++)
  {
    const int32_t offset_m = (block_mid * BLOCK_SIZE_M + m * 16 + (lane_id & 15));
    inline_buffer_load_dword(token_index_arr_new[m], offset_m, g_sorted_token_ids, 0);
  }

  for (int m = 0; m < (BLOCK_SIZE_M + 15) / 16; m++)
  {
    for (int m_id = 0; m_id < (nums_token_a_warp_load + 1) / 2; m_id++)
    {
      const int32_t offset_m = block_mid * BLOCK_SIZE_M + m * 16 + (warp_id * nums_token_a_warp_load + (lane_id / 32) + m_id * 2) /* > 15 ? 15: (warp_id*4 + (lane_id / 32) + m_id * 2) */;
      inline_buffer_load_dword(token_index_arr_32[m][m_id], offset_m, g_sorted_token_ids, 0);
    }
  }
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0) \n\t"
               "s_barrier");
  __builtin_amdgcn_sched_barrier(0);

  __syncthreads();

  _Pragma("unroll 1") for (int k_loop = 0; k_loop < k_loop_num; k_loop++)
  {
    int offset_k = k_loop * BLOCK_SIZE_K + block_kid * BLOCK_SIZE_K * BLOCK_SIZE_K_LOOPS;

    /*左矩阵 load - lds BLOCK_SIZE_K must be equal to 128*/
    _Pragma("unroll") for (int m = 0; m < (BLOCK_SIZE_M + 15) / 16; m++)
    {
      _Pragma("unroll") for (int m_id = 0; m_id < nums_token_a_warp_load / 2; m_id++)
      {
        int padding = warp_id * 2 + m_id + m * 8;
        int origin_k;
        origin_k = (lane_id & 31) * 4; /* 修改为*4 */
        int token_index = token_index_arr_32[m][m_id];
        if ((token_index / top_k < size_m) && (m * 16 + (nums_token_a_warp_load)*warp_id + m_id * 2 + lane_id / 32) < BLOCK_SIZE_M)
        {
          int lds_offset = __builtin_amdgcn_readfirstlane(m * 16 * BLOCK_SIZE_K + (warp_id * (nums_token_a_warp_load) + m_id * 2) * BLOCK_SIZE_K) / 4 /* 修改为/4 */ + padding;
          origin_k += token_index / top_k * size_k + offset_k;
          inline_buffer_load_dword_lds(block_input_vec4_int8, g_input, lds_offset, 0, origin_k / 4 /* 修改为/4 */);
        }
      }
    }

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0) \n\t"
                 "s_barrier");
    __builtin_amdgcn_sched_barrier(0);

    __syncthreads();

    auto g_expert_scales_input = tcp_cache_swizzle_func<1, float>(expert_scales_input);
    /*左矩阵提前到这里空出lds（lds直接读到register）的使用给酉矩阵（16k）提高并行度（4个blocks）   */
    _Pragma("unroll") for (int tmp_k = 0; tmp_k < (BLOCK_SIZE_K / 32 /* 修改为/32 */); tmp_k++)
    {
      _Pragma("unroll") for (int m = 0; m < ((BLOCK_SIZE_M + 15) / 16) /*num_valid_tokens*/; m++)
      { /*有效的token数量*/
        if ((token_index_arr_new[m] / top_k < size_m) && ((m * 16 + (lane_id & 15)) < BLOCK_SIZE_M))
        {
          {
            int padding = m * 8 + (lane_id & 15) / 2;
            int offset_input = ((m * 16 + (lane_id & 15)) * BLOCK_SIZE_K + tmp_k * 32 /* 修改为*32 */ + (lane_id >> 4) * 8) / 4 + padding;
            input_vec8_int8[m][tmp_k].i32[0] = block_input_vec4_int8[offset_input]; /* 接受数据的寄存器数据类型需要修改 */
            input_vec8_int8[m][tmp_k].i32[1] = block_input_vec4_int8[offset_input + 1];
          }
        }
      }
    }

    _Pragma("unroll") for (int m = 0; m < ((BLOCK_SIZE_M + 15) / 16) /*num_valid_tokens*/; m++)
    { /*有效的token数量*/
      int group_id_m = token_index_arr_new[m] / top_k;
      int group_id_k = (offset_k) / group_size_k;
      int scales_offset_tmp_input = group_id_m * (size_k / group_size_k) + group_id_k;
      if ((token_index_arr_new[m] / top_k < size_m) && ((m * 16 + (lane_id & 15)) < BLOCK_SIZE_M))
      {

        inline_buffer_load_dword(input_scales_float[m], scales_offset_tmp_input, g_expert_scales_input, 0);
      }
      __syncthreads();
      __builtin_amdgcn_sched_barrier(0);
      asm volatile("s_waitcnt vmcnt(0) \n\t"
                   "s_barrier");
      __builtin_amdgcn_sched_barrier(0);
    }

    int stage_id = 0;
    /* if(n_loop_num == 1){
      nloop(0);
    }else if(n_loop_num == 2){
      nloop(0);
      nloop(1);
    }else if(n_loop_num == 4){
      nloop(0);
      nloop(1);
      nloop(2);
      nloop(3);
    } */

    auto func_nloop = [&](int n_loop)
    {
      /* int k_loop = 0; */
      /* blockIdx.x * BLOCK_SIZE_K*BLOCK_SIZE_K_LOOPS + k_loop*BLOCK_SIZE_K; */
      float expert_scales_groups_float;

      int scale_zero_offset_n = block_nid * BLOCK_SIZE_N * BLOCK_SIZE_N_LOOPS + n_loop * BLOCK_SIZE_N + warp_id * WARP_N + laneid_and_15 /*((WARP_N/16 == 2) ? (laneid_and_15*2):(laneid_and_15))*/;
      int scale_zero_offset_m = block_mid * BLOCK_SIZE_M + laneid_and_15;

      auto g_expert_scales = tcp_cache_swizzle_func<1, float>(expert_scales);

      __builtin_amdgcn_sched_barrier(0);
      for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
      {
        for (int j = 0; j < (WARP_N / 16); j++)
        { /*奇偶分开*/
          int group_id_n = (scale_zero_offset_n + n_idx * 32 + j * 16) / group_size_n;
          int group_id_k = (offset_k) / group_size_k;
          int scales_offset_tmp_weight = group_id_n * (size_k / group_size_k) + group_id_k;

          inline_buffer_load_dword(expert_scales_groups[n_idx][j].float_array[0], scales_offset_tmp_weight, g_expert_scales, 0);
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      __builtin_amdgcn_sched_barrier(0);
      asm volatile("s_waitcnt vmcnt(0) \n\t"
                   "s_barrier");
      __builtin_amdgcn_sched_barrier(0);

      for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
      {
        for (int j = 0; j < (WARP_N / 16); j++)
        { /*奇偶分开*/
          expert_scales_groups_float = (expert_scales_groups[n_idx][j].float_array[0]);
        }
      }
      for (int m = 0; m < ((BLOCK_SIZE_M + 15) / 16) /*num_valid_tokens*/; m++)
      {
        final_scales[m] = expert_scales_groups_float * input_scales_float[m];
      }

      if (n_loop == 0)
      {
        _Pragma("unroll") /* 类大kloop循环开始 */
            for (int tmp_k1 = 0; tmp_k1 < BLOCK_SIZE_K / K_TILE; tmp_k1++)
        {

          __builtin_amdgcn_sched_barrier(0);

          { /*stage 1 load data*/
            constexpr int W_LOAD_REQUESTS = WARP_N / 4;
            int offset_n = block_nid * BLOCK_SIZE_N_LOOPS * BLOCK_SIZE_N + n_loop * BLOCK_SIZE_N + w_lane_n_idx;
            int k = offset_k + tmp_k1 * 64 + w_lane_k_idx * 4;
            const int32_t weight_offset = offset_n * size_k + k;
            int w_lds_stage_offset = stage_id * BLOCK_SIZE_N * BLOCK_SIZE_K;

            {

              __builtin_amdgcn_sched_barrier(0);
              _Pragma("unroll") for (int load = 0; load < W_LOAD_REQUESTS; ++load)
              {
                int padding = 0; /*padding size in shared memory per buffer load, to avoid bank conflict*/
                int w_warp_buffer_load_lds_offset = w_lds_stage_offset + warp_id * (WARP_N * 64) + load * (4 * 64) + tmp_k1 * (BLOCK_SIZE_N / WARP_N) * (K_TILE / 64) * (WARP_N * 64);

                int gvOffset_s = 0;
                int gvOffset_v = (weight_offset + (warp_id * WARP_N + load * 4) * size_k) / pack_factor;

                int lds_offset = __builtin_amdgcn_readfirstlane(w_warp_buffer_load_lds_offset + padding) / pack_factor;
                inline_buffer_load_dword_lds(weight_lds, g_expert_qweight, lds_offset, gvOffset_s, gvOffset_v);
              }
              __builtin_amdgcn_sched_barrier(0);
            }

          } /*global-> lds (copy from fa)*/
        } /* fetch data finished */
      }
      __builtin_amdgcn_sched_barrier(0);

      if (n_loop != BLOCK_SIZE_N_LOOPS - 1)
      {
        stage_id ^= 1;
        _Pragma("unroll") /* 类大kloop循环开始 */
            for (int tmp_k1 = 0; tmp_k1 < BLOCK_SIZE_K / K_TILE; tmp_k1++)
        {

          __builtin_amdgcn_sched_barrier(0);

          { /*stage 1 load data*/
            constexpr int W_LOAD_REQUESTS = WARP_N / 4;
            int offset_n = block_nid * BLOCK_SIZE_N_LOOPS * BLOCK_SIZE_N + (n_loop + 1) * BLOCK_SIZE_N + w_lane_n_idx;
            int k = offset_k + tmp_k1 * 64 + w_lane_k_idx * 4;
            const int32_t weight_offset = offset_n * size_k + k;
            int w_lds_stage_offset = stage_id * BLOCK_SIZE_N * BLOCK_SIZE_K;

            {

              __builtin_amdgcn_sched_barrier(0);
              _Pragma("unroll") for (int load = 0; load < W_LOAD_REQUESTS; ++load)
              {
                int padding = 0; /*padding size in shared memory per buffer load, to avoid bank conflict*/
                int w_warp_buffer_load_lds_offset = w_lds_stage_offset + warp_id * (WARP_N * 64) + load * (4 * 64) + tmp_k1 * (BLOCK_SIZE_N / WARP_N) * (K_TILE / 64) * (WARP_N * 64);

                int gvOffset_s = 0;
                int gvOffset_v = (weight_offset + (warp_id * WARP_N + load * 4) * size_k) / pack_factor;

                int lds_offset = __builtin_amdgcn_readfirstlane(w_warp_buffer_load_lds_offset + padding) / pack_factor;
                inline_buffer_load_dword_lds(weight_lds, g_expert_qweight, lds_offset, gvOffset_s, gvOffset_v);
              }
              __builtin_amdgcn_sched_barrier(0);
            }

          } /*global-> lds (copy from fa)*/
        } /* fetch data finished */

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(4) \n\t"
                     "s_barrier");
        __builtin_amdgcn_sched_barrier(0);
      }
      else
      {
        stage_id ^= 1;
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0) \n\t"
                     "s_barrier");
        __builtin_amdgcn_sched_barrier(0);
      }

      stage_id ^= 1;
      _Pragma("unroll") for (int tmp_k1 = 0; tmp_k1 < BLOCK_SIZE_K / K_TILE; tmp_k1++)
      {

        {

          {
            /*lds -> vgpr use ds_read_m; right matrix*/

            int w_lds_stage_offset = stage_id * BLOCK_SIZE_N * BLOCK_SIZE_K;

            int *weight_v4int8 = (int *)(weight_lds);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            _Pragma("unroll") for (int tmp_k2 = 0; tmp_k2 < (K_TILE / 64); tmp_k2++)
            { /*32 half in col direction*/
              _Pragma("unroll") for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
              {
                _Pragma("unroll") for (int i = 0; i < (WARP_N / 16); i++)
                { /*n direction*/
                  _Pragma("unroll") for (int j = 0; j < (64 / 32); j++)
                  { /*k direction,K_TILE=64, (t0,t16,t32,t48) load 16 int8 data per loop*/
                    int lds_offset = (w_lds_stage_offset + tmp_k1 * (BLOCK_SIZE_N / WARP_N) * (K_TILE / 64) * (WARP_N * 64) + warp_id * (WARP_N * 64) + ((lane_id & 15) * 64) + j * 32 + (lane_id / 16) * 8) / 4;
                    inline_ds_read_b32_no_wait<int>(weight_v4int8, lds_offset, expert_w_reg[tmp_k1][i][j].i32[0]);
                    inline_ds_read_b32_no_wait<int>(weight_v4int8, lds_offset + 1, expert_w_reg[tmp_k1][i][j].i32[1]); /* expert_w_reg是指向结构体指针 使用方法不一定正确 */
                  }
                }
              }
            }
            asm volatile("s_setprio 0");
            __builtin_amdgcn_sched_barrier(0);
          }

        } /*ds-read: register read data from lds*/
      }

      /*lds -> vgpr use ds_read_m; right matrix*/

      __builtin_amdgcn_sched_barrier(0);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_sched_barrier(0);
      _Pragma("unroll") for (int tmp_k1 = 0; tmp_k1 < BLOCK_SIZE_K / K_TILE; tmp_k1++)
      {
        /*======================================= mmac compute =============================================*/
        {
          __builtin_amdgcn_sched_barrier(0);

          _Pragma("unroll") for (int m = 0; m < ((BLOCK_SIZE_M + 15) / 16) /*num_valid_tokens*/; m++)
          {

            asm volatile("s_setprio 1");

            _Pragma("unroll") for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
            {
              for (int k_idx = 0; k_idx < (K_TILE / 32); k_idx++)
              {
                /*min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16*/
                _Pragma("unroll") for (int min_tile_n = 0; min_tile_n < (WARP_N / 16); min_tile_n++)
                {

                  res_tmp[m][n_idx * (WARP_N / 16) + min_tile_n] =
                      __builtin_hcu_mmac_i32_16x16x32_i8(vec8_int8{
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[0],
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[1],
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[2],
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[3],
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[4],
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[5],
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[6],
                                                             input_vec8_int8[m][k_idx + (tmp_k1) * (K_TILE / 32)].i8[7]},
                                                         vec8_int8{expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[0], expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[1], expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[2], expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[3], expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[4], expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[5], expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[6], expert_w_reg[tmp_k1][min_tile_n][k_idx].i8[7]}, res_zeros);

                  __builtin_amdgcn_sched_barrier(0);
                  asm volatile("s_waitcnt vmcnt(0)");
                  __builtin_amdgcn_sched_barrier(0);

                  for (int i = 0; i < 4; i++)
                  {
                    res[m][n_idx * (WARP_N / 16) + min_tile_n][n_loop][i] += res_tmp[m][n_idx * (WARP_N / 16) + min_tile_n][i] * final_scales[m] /* input_scales_float[m] * expert_scales_groups_float */;
                  }
                }
              }
            }
            asm volatile("s_setprio 0");
          }
          __syncthreads();
          __builtin_amdgcn_sched_barrier(0);
        }
      }
      stage_id ^= 1;
    };

    if (BLOCK_SIZE_N_LOOPS == 1)
    {
      func_nloop(0);
    }
    else if (BLOCK_SIZE_N_LOOPS == 2)
    {
      func_nloop(0);
      func_nloop(1);
    }
    else if (BLOCK_SIZE_N_LOOPS == 4)
    {
      func_nloop(0);
      func_nloop(1);
      func_nloop(2);
      func_nloop(3);
    }
  }

  /*======================= output ======================*/

  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)");
  __builtin_amdgcn_sched_barrier(0);
  __syncthreads();

  for (int n_loop = 0; n_loop < BLOCK_SIZE_N_LOOPS; n_loop++)
  {
    int offset_n = block_nid * BLOCK_SIZE_N_LOOPS * BLOCK_SIZE_N + n_loop * BLOCK_SIZE_N + warp_id * WARP_N + laneid_shfl_4 * (WARP_N / 16);

    {
      /* 加反量化 */

      _Pragma("unroll") for (int m = 0; m < ((BLOCK_SIZE_M + 15) / 16) /*num_valid_tokens*/; m++)
      {
        _Pragma("unroll") for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
        {
          _Pragma("unroll") for (int i = 0; i < (WARP_N / 16); i++)
          {
            _Pragma("unroll") for (int j = 0; j < 4; j++)
            { /*k direction,K_TILE=64, (t0,t16,t32,t48) load 16 int8 data per loop*/
              res_float[m][n_idx * (WARP_N / 16) + i][n_loop][j] = res[m][n_idx * (WARP_N / 16) + i][n_loop][j] /* * expert_scales_groups_input[m] * expert_scales_groups[0][0].float_array[0] */;
            }
          }
        }
      }

      __builtin_amdgcn_sched_barrier(0);
      asm volatile("s_waitcnt vmcnt(0)");
      __builtin_amdgcn_sched_barrier(0);
      _Pragma("unroll") for (int m = 0; m < (BLOCK_SIZE_M + 15) / 16 /*num_valid_tokens*/; ++m)
      {
        int32_t token_index = token_index_arr_new[m];
        if ((token_index / top_k < size_m) && ((m * 16 + (lane_id & 15)) < BLOCK_SIZE_M))
        {

          /*乘以对应expert的权重*/
          if (mul_topk_weight)
          {
            _Pragma("unroll") for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
            {
              _Pragma("unroll") for (int min_tile_n = 0; min_tile_n < (WARP_N / 16); min_tile_n++)
              {
                _Pragma("unroll") for (int vec_id = 0; vec_id < 4; vec_id++)
                {
                  res_float[m][n_idx * ((WARP_N + 31) / 32) + min_tile_n][n_loop][vec_id] *= topk_weights[token_index];
                }
              }
            }
          }
          __builtin_amdgcn_sched_barrier(0);
          asm volatile("s_waitcnt vmcnt(0)");
          __builtin_amdgcn_sched_barrier(0);
          /*if(k_loop == (BLOCK_SIZE_K_LOOPS - 1)) */
          {
            if (USE_ATOMIC)
            {
              _Pragma("unroll") for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
              {
                _Pragma("unroll") for (int min_tile_n = 0; min_tile_n < (WARP_N / 16); min_tile_n++)
                {
                  _Pragma("unroll") for (int vec_id = 0; vec_id < 4; vec_id++)
                  {
                    __builtin_hcu_global_atomic_fadd_f32(&output[token_index * size_n + n_idx * 32 + offset_n + min_tile_n * 16 + vec_id * 4 * (WARP_N / 16)], res_float[m][n_idx * ((WARP_N + 31) / 32) + min_tile_n][n_loop][vec_id]);
                  }
                }
              }
            }
            else
            {
              half_t *output_fp16 = reinterpret_cast<half_t *>(output);
              _Pragma("unroll") for (int n_idx = 0; n_idx < ((WARP_N + 31) / 32); n_idx++)
              {
                _Pragma("unroll") for (int min_tile_n = 0; min_tile_n < (WARP_N / 16); min_tile_n++)
                {
                  _Pragma("unroll") for (int vec_id = 0; vec_id < 4; vec_id++)
                  {
                    output_fp16[token_index * size_n + offset_n + n_idx * 32 + min_tile_n * 16 + vec_id * 4 * (WARP_N / 16)] = __float2half(res_float[m][n_idx * ((WARP_N + 31) / 32) + min_tile_n][n_loop][vec_id]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  __syncthreads();

  /* if(i<422) goto start; */

} // kernel

template <typename scalar_t,
          int bit,
          int top_k,
          int BLOCK_SIZE_M,
          int BLOCK_SIZE_N,
          int BLOCK_SIZE_K,
          bool has_zp,
          bool mul_topk_weight,
          int group_size_n,
          int group_size_k,
          int BLOCK_SIZE_M_LOOPS,
          int BLOCK_SIZE_N_LOOPS,
          int BLOCK_SIZE_K_LOOPS,
          bool USE_ATOMIC,
          int THREADS_PER_BLOCK>
void run_moe_w8a8_gemm_block_wise_kernel2(const uint32_t *input,
                                          const float *a_scales,
                                          float *output_fp32,
                                          const uint32_t *b_qweight,
                                          int *b_qweight_out,
                                          const float *b_scales,
                                          const uint32_t *b_qzeros,
                                          const float *topk_weights,
                                          const int32_t *sorted_token_ids,
                                          const int32_t *expert_ids,
                                          const int32_t *num_tokens_post_pad,
                                          int num_token_blocks,
                                          int size_m,
                                          int size_n,
                                          int size_k)
{
  dim3 blockDim, gridDim;
  int WARP_N = 16;
  blockDim.x = (BLOCK_SIZE_N / WARP_N) * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  gridDim.z = min(size_m * top_k, num_token_blocks);
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N * BLOCK_SIZE_N_LOOPS);
  gridDim.x = DIVIDE(size_k, BLOCK_SIZE_K * BLOCK_SIZE_K_LOOPS);

  auto kernel = moe_w8a8_gemm_kernel_block_wise_kernel2<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size_n, group_size_k, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC, THREADS_PER_BLOCK>;
  constexpr int K_TILE = 64;
  const int shared_mem_size = std::max(BLOCK_SIZE_M_LOOPS * (BLOCK_SIZE_M + 15) / 16 * 16 * BLOCK_SIZE_K, BLOCK_SIZE_K * BLOCK_SIZE_N * 2);

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  kernel<<<gridDim, blockDim, shared_mem_size, stream>>>(
      input,
      a_scales,
      output_fp32,
      b_qweight,
      b_qweight_out,
      b_scales,
      b_qzeros,
      topk_weights,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      num_token_blocks,
      size_m, size_n, size_k); //, block_size_k_loops);
}
