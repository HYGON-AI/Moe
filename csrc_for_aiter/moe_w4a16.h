
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "moe_wna16_utils.h"

#include "wait.h"
#include "intrinsic.h"

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#define int4_moe_wna16_cal(n_loop)                                                                                                                                                        \
  {                                                                                                                                                                                       \
      /*k_loop == 0 */                                                                                                                                                                    \
      {                                                                                                                                                                                   \
          const int32_t offset_k = blockIdx.x * BLOCK_SIZE_K * BLOCK_SIZE_K_LOOPS + k_loop * BLOCK_SIZE_K;                                                                                \
  if (n_loop == 0)                                                                                                                                                                        \
  {                                                                                                                                                                                       \
                                                                                                                                                                                          \
    stage_id = 0;                                                                                                                                                                         \
    for (int m = 0; m < BLOCK_SIZE_M / 16; m++)                                                                                                                                           \
    {                                                                                                                                                                                     \
      int32_t offset_m = blockIdx.z * BLOCK_SIZE_M + m * 16 / WARP_NUM + warp_id * BLOCK_SIZE_M / WARP_NUM;                                                                               \
      auto g_s_sorted_token_ids = reinterpret_cast<uint64_t>(sorted_token_ids + offset_m);                                                                                                \
      inline_s_load_dwordx4(token_index_arr.vec4_i32[m], g_s_sorted_token_ids, 0);                                                                                                        \
    }                                                                                                                                                                                     \
                                                                                                                                                                                          \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
    asm volatile("s_waitcnt lgkmcnt(0)");                                                                                                                                                 \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
                                                                                                                                                                                          \
    for (int m = 0; m < BLOCK_SIZE_M / 16; m++)                                                                                                                                           \
    {                                                                                                                                                                                     \
      int32_t offset_m = (blockIdx.z * BLOCK_SIZE_M + m * 16 + (lane_id & 15));                                                                                                           \
      inline_buffer_load_dword(token_index_arr_new[m], offset_m, g_sorted_token_ids, 0);                                                                                                  \
    }                                                                                                                                                                                     \
                                                                                                                                                                                          \
    /*左矩阵 load - lds BLOCK_SIZE_K must be equal to 128*/                                                                                                                               \
    _Pragma("unroll") for (int m_id = 0; m_id < (BLOCK_SIZE_M + WARP_NUM - 1) / WARP_NUM; m_id++)                                                                                         \
    {                                                                                                                                                                                     \
      int origin_k;                                                                                                                                                                       \
      origin_k = lane_id * 2;                                                                                                                                                             \
      const int32_t token_index = token_index_arr.i32[m_id];                                                                                                                              \
      if (token_index / top_k < size_m)                                                                                                                                                   \
      {                                                                                                                                                                                   \
        int lds_offset = __builtin_amdgcn_readfirstlane((warp_id * (BLOCK_SIZE_M / WARP_NUM) + m_id) * BLOCK_SIZE_K / 2 + (warp_id * ((BLOCK_SIZE_M + WARP_NUM - 1) / WARP_NUM) + m_id)); \
        origin_k += token_index / top_k * size_k + blockIdx.x * BLOCK_SIZE_K_LOOPS * BLOCK_SIZE_K + k_loop * BLOCK_SIZE_K;                                                                \
        inline_buffer_load_dword_lds(block_input, g_input, lds_offset, 0, origin_k / 2);                                                                                                  \
      }                                                                                                                                                                                   \
    }                                                                                                                                                                                     \
                                                                                                                                                                                          \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
    asm volatile("s_waitcnt vmcnt(0) \n\t"                                                                                                                                                \
                 "s_barrier");                                                                                                                                                            \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
                                                                                                                                                                                          \
    /*左矩阵提前到这里空出lds（lds直接读到register）的使用给酉矩阵（16k）提高并行度（4个blocks）   */                                                                                     \
    _Pragma("unroll") for (int tmp_k = 0; tmp_k < (BLOCK_SIZE_K / 32); tmp_k++)                                                                                                           \
    {                                                                                                                                                                                     \
      _Pragma("unroll") for (int m = 0; m < ((BLOCK_SIZE_M + 15) / 16) /*num_valid_tokens*/; m++)                                                                                         \
      { /*有效的token数量*/                                                                                                                                                               \
        if (token_index_arr_new[m] / top_k < size_m)                                                                                                                                      \
        {                                                                                                                                                                                 \
          {                                                                                                                                                                               \
            int offset_input = ((m * 16 + (lane_id & 15)) * (BLOCK_SIZE_K + 2) + tmp_k * 32 + (lane_id >> 4) * 8) / 2;                                                                    \
            input_vec8_fp16[m][tmp_k].input_half2[0] = block_input_half2[offset_input];                                                                                                   \
            input_vec8_fp16[m][tmp_k].input_half2[1] = block_input_half2[offset_input + 1];                                                                                               \
            input_vec8_fp16[m][tmp_k].input_half2[2] = block_input_half2[offset_input + 2];                                                                                               \
            input_vec8_fp16[m][tmp_k].input_half2[3] = block_input_half2[offset_input + 3];                                                                                               \
          }                                                                                                                                                                               \
        }                                                                                                                                                                                 \
      }                                                                                                                                                                                   \
    }                                                                                                                                                                                     \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
    asm volatile("s_waitcnt lgkmcnt(0) \n\t"                                                                                                                                              \
                 "s_barrier");                                                                                                                                                            \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
                                                                                                                                                                                          \
    { /*stage 1 load data*/                                                                                                                                                               \
      constexpr int W_LOAD_REQUESTS = WARP_N / 4;                                                                                                                                         \
      int offset_n = blockIdx.y * BLOCK_SIZE_N_LOOPS * BLOCK_SIZE_N + n_loop * BLOCK_SIZE_N + lane_id;                                                                                    \
      const int32_t weight_offset = offset_n + (blockIdx.x * BLOCK_SIZE_K_LOOPS + k_loop) * BLOCK_SIZE_K * size_n * 256 / 8 + warp_id * size_n;                                           \
                                                                                                                                                                                          \
      {                                                                                                                                                                                   \
                                                                                                                                                                                          \
        __builtin_amdgcn_sched_barrier(0);                                                                                                                                                \
        _Pragma("unroll") for (int load = 0; load < W_LOAD_REQUESTS; ++load)                                                                                                              \
        {                                                                                                                                                                                 \
          int w_warp_buffer_load_lds_offset = warp_id * 80 * 8 + load * 4 * 80 * 8;                                                                                                       \
          int gvOffset_s = 0;                                                                                                                                                             \
          lds_stage_offset = stage_id * (80 * BLOCK_SIZE_K / 8);                                                                                                                          \
          int gvOffset_v = (weight_offset + load * 4 * size_n);                                                                                                                           \
          int lds_offset = __builtin_amdgcn_readfirstlane(w_warp_buffer_load_lds_offset / pack_factor + lds_stage_offset);                                                                \
          inline_buffer_load_dword_lds(weight_lds, g_expert_qweight, lds_offset, gvOffset_s, gvOffset_v);                                                                                 \
        }                                                                                                                                                                                 \
        __builtin_amdgcn_sched_barrier(0);                                                                                                                                                \
      }                                                                                                                                                                                   \
                                                                                                                                                                                          \
    } /*global-> lds (copy from fa)*/                                                                                                                                                     \
                                                                                                                                                                                          \
    stage_id = stage_id ^ 1;                                                                                                                                                              \
  }                                                                                                                                                                                       \
                                                                                                                                                                                          \
  int scale_offset_n = blockIdx.y * BLOCK_SIZE_N * BLOCK_SIZE_N_LOOPS + n_loop * BLOCK_SIZE_N + warp_id * WARP_N + laneid_and_15;                                                         \
                                                                                                                                                                                          \
  int zero_offset_n = (blockIdx.y * BLOCK_SIZE_N * BLOCK_SIZE_N_LOOPS + n_loop * BLOCK_SIZE_N + warp_id * WARP_N + laneid_and_15) / 2;                                                    \
                                                                                                                                                                                          \
  auto g_expert_scales = tcp_cache_swizzle_func<1, scalar_t>(expert_scales);                                                                                                              \
                                                                                                                                                                                          \
  {                                                                                                                                                                                       \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
    int scales_offset_tmp =                                                                                                                                                               \
        (scale_offset_n * size_k + offset_k) / group_size / GROUPS;                                                                                                                       \
    if constexpr (GROUPS == 2)                                                                                                                                                            \
    {                                                                                                                                                                                     \
      inline_buffer_load_dword<float, 2>(expert_scales_groups.float_array[0], scales_offset_tmp, g_expert_scales, 0);                                                                     \
    }                                                                                                                                                                                     \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
                                                                                                                                                                                          \
    if (has_zp)                                                                                                                                                                           \
    {                                                                                                                                                                                     \
      if constexpr (GROUPS == 2)                                                                                                                                                          \
      {                                                                                                                                                                                   \
        int qzeros_offset_tmp =                                                                                                                                                           \
            (zero_offset_n) * (size_k / group_size) +                                                                                                                                     \
            offset_k / group_size;                                                                                                                                                        \
        inline_buffer_load_ubyte<uint32_t>(uint32_zero_tmp[0], qzeros_offset_tmp, g_expert_qzero, 0);                                                                                     \
        inline_buffer_load_ubyte<uint32_t>(uint32_zero_tmp[1], qzeros_offset_tmp + 1, g_expert_qzero, 0);                                                                                 \
      }                                                                                                                                                                                   \
    }                                                                                                                                                                                     \
  }                                                                                                                                                                                       \
                                                                                                                                                                                          \
  if (n_loop != (BLOCK_SIZE_N_LOOPS - 1))                                                                                                                                                 \
  { /*stage 1 load data*/                                                                                                                                                                 \
    constexpr int W_LOAD_REQUESTS = WARP_N / 4;                                                                                                                                           \
    int offset_n = blockIdx.y * BLOCK_SIZE_N_LOOPS * BLOCK_SIZE_N + (n_loop + 1) * BLOCK_SIZE_N + lane_id;                                                                                \
    const int32_t weight_offset = offset_n + (blockIdx.x * BLOCK_SIZE_K_LOOPS + k_loop) * BLOCK_SIZE_K * size_n * 256 / 8 + warp_id * size_n;                                             \
                                                                                                                                                                                          \
    {                                                                                                                                                                                     \
                                                                                                                                                                                          \
      __builtin_amdgcn_sched_barrier(0);                                                                                                                                                  \
      _Pragma("unroll") for (int load = 0; load < W_LOAD_REQUESTS; ++load)                                                                                                                \
      {                                                                                                                                                                                   \
        int w_warp_buffer_load_lds_offset = warp_id * 80 * 8 + load * 4 * 80 * 8;                                                                                                         \
        int gvOffset_s = 0;                                                                                                                                                               \
        lds_stage_offset = stage_id * (80 * BLOCK_SIZE_K / 8);                                                                                                                            \
        int gvOffset_v = (weight_offset + load * 4 * size_n);                                                                                                                             \
        int lds_offset = __builtin_amdgcn_readfirstlane(w_warp_buffer_load_lds_offset / pack_factor + lds_stage_offset);                                                                  \
        inline_buffer_load_dword_lds(weight_lds, g_expert_qweight, lds_offset, gvOffset_s, gvOffset_v);                                                                                   \
      }                                                                                                                                                                                   \
      __builtin_amdgcn_sched_barrier(0);                                                                                                                                                  \
    }                                                                                                                                                                                     \
                                                                                                                                                                                          \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
    asm volatile("s_waitcnt vmcnt(4) \n\t"                                                                                                                                                \
                 "s_barrier");                                                                                                                                                            \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
                                                                                                                                                                                          \
  } /*global-> lds (copy from fa)*/                                                                                                                                                       \
  else                                                                                                                                                                                    \
  {                                                                                                                                                                                       \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
    asm volatile("s_waitcnt vmcnt(0) \n\t"                                                                                                                                                \
                 "s_barrier");                                                                                                                                                            \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
  }                                                                                                                                                                                       \
                                                                                                                                                                                          \
  { /*stage 0 , ds read*/                                                                                                                                                                 \
                                                                                                                                                                                          \
    stage_id = stage_id ^ 1;                                                                                                                                                              \
    {                                                                                                                                                                                     \
      {                                                                                                                                                                                   \
        /*lds -> vgpr use ds_read_m; right matrix*/                                                                                                                                       \
                                                                                                                                                                                          \
        int *weight_v4int8 = (int *)(weight_lds);                                                                                                                                         \
        __builtin_amdgcn_sched_barrier(0);                                                                                                                                                \
        asm volatile("s_setprio 1");                                                                                                                                                      \
        _Pragma("unroll") for (int j = 0; j < (128 / 32); j++)                                                                                                                            \
        { /*k direction,K_TILE=64, (t0,t16,t32,t48) load 16 int8 data per loop*/                                                                                                          \
          int lds_offset = (warp_id * 16 * 8 + ((lane_id & 15) * 8) + j * 4 * 8 * 80 + (lane_id / 16) * 8 * 80) / pack_factor + stage_id * (80 * K_TILE / 8);                             \
          inline_ds_read_b32_no_wait<int>(weight_v4int8, lds_offset, expert_w_reg[j].i32);                                                                                                \
        }                                                                                                                                                                                 \
        asm volatile("s_setprio 0");                                                                                                                                                      \
        __builtin_amdgcn_sched_barrier(0);                                                                                                                                                \
      }                                                                                                                                                                                   \
                                                                                                                                                                                          \
    } /*ds-read: register read data from lds*/                                                                                                                                            \
                                                                                                                                                                                          \
    reg_bf16_fp16<scalar_t2> k_reg_half2[4];                                                                                                                                              \
                                                                                                                                                                                          \
    /*lds -> vgpr use ds_read_m; right matrix*/                                                                                                                                           \
                                                                                                                                                                                          \
    if (has_zp)                                                                                                                                                                           \
    {                                                                                                                                                                                     \
      if constexpr (GROUPS == 2)                                                                                                                                                          \
      {                                                                                                                                                                                   \
        if ((lane_id & 1) == 0)                                                                                                                                                           \
        {                                                                                                                                                                                 \
          expert_qzeros_groups.uint8_array[0] = uint32_zero_tmp[0] & 0xF;                                                                                                                 \
          expert_qzeros_groups.uint8_array[1] = uint32_zero_tmp[1] & 0xF;                                                                                                                 \
        }                                                                                                                                                                                 \
        else                                                                                                                                                                              \
        {                                                                                                                                                                                 \
          expert_qzeros_groups.uint8_array[0] = (uint32_zero_tmp[0] >> 4) & 0xF;                                                                                                          \
          expert_qzeros_groups.uint8_array[1] = (uint32_zero_tmp[1] >> 4) & 0xF;                                                                                                          \
        }                                                                                                                                                                                 \
      }                                                                                                                                                                                   \
    }                                                                                                                                                                                     \
    {                                                                                                                                                                                     \
      /*q_scale和qzero每个group_size一个*/                                                                                                                                                \
      {                                                                                                                                                                                   \
        scalar_t scale_f_0 = expert_scales_groups.scalar_array[0];                                                                                                                        \
        scale_f2[0] = Dtype::num2num2(scale_f_0);                                                                                                                                         \
        scalar_t scale_f_1 = expert_scales_groups.scalar_array[1];                                                                                                                        \
        scale_f2[1] = Dtype::num2num2(scale_f_1);                                                                                                                                         \
        if (has_zp)                                                                                                                                                                       \
        {                                                                                                                                                                                 \
          uint8_t qzero_0 = expert_qzeros_groups.uint8_array[0];                                                                                                                          \
          qzero_f2[0] = Dtype::num2num2(Dtype::int2num(qzero_0));                                                                                                                         \
          uint8_t qzero_1 = expert_qzeros_groups.uint8_array[1];                                                                                                                          \
          qzero_f2[1] = Dtype::num2num2(Dtype::int2num(qzero_1));                                                                                                                         \
        }                                                                                                                                                                                 \
        else                                                                                                                                                                              \
        {                                                                                                                                                                                 \
          qzero_f2[0] = Dtype::num2num2(Dtype::int2num(128));                                                                                                                             \
        }                                                                                                                                                                                 \
      } /*scale and zero data load*/                                                                                                                                                      \
    } /*register -> register*/                                                                                                                                                            \
                                                                                                                                                                                          \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
    asm volatile("s_waitcnt lgkmcnt(0) \n\t"                                                                                                                                              \
                 "s_barrier");                                                                                                                                                            \
    __builtin_amdgcn_sched_barrier(0);                                                                                                                                                    \
                                                                                                                                                                                          \
    _Pragma("unroll") for (int j = 0; j < K_TILE / 32; j++)                                                                                                                               \
    { /*k direction,K_TILE=64, (t0,t16,t32,t48) load 16 int8 data per loop*/                                                                                                              \
      dequant<scalar_t2, bit>(expert_w_reg[j].i32, k_reg_half2[j].bf162);                                                                                                                 \
    }                                                                                                                                                                                     \
    _Pragma("unroll") for (int j = 0; j < 4; j++)                                                                                                                                         \
    { /*k direction,K_TILE=128, (t0,t16,t32,t48) load 32 int4 data per loop*/                                                                                                             \
      _Pragma("unroll") for (int pack_id = 0; pack_id < 16 / bit; pack_id++)                                                                                                              \
      {                                                                                                                                                                                   \
        k_reg_half2[j].bf162[pack_id] = __hmul2(__hsub2(k_reg_half2[j].bf162[pack_id], qzero_f2[j / 2]), scale_f2[j / 2]);                                                                \
      }                                                                                                                                                                                   \
    }                                                                                                                                                                                     \
                                                                                                                                                                                          \
    /*======================================= mmac compute =============================================*/                                                                                \
    {                                                                                                                                                                                     \
      __builtin_amdgcn_sched_barrier(0);                                                                                                                                                  \
                                                                                                                                                                                          \
      _Pragma("unroll") for (int m = 0; m < ((BLOCK_SIZE_M + 15) / 16) /*num_valid_tokens*/; m++)                                                                                         \
      {                                                                                                                                                                                   \
                                                                                                                                                                                          \
        asm volatile("s_setprio 1");                                                                                                                                                      \
                                                                                                                                                                                          \
        for (int k_idx = 0; k_idx < (K_TILE / 32); k_idx++)                                                                                                                               \
        {                                                                                                                                                                                 \
                                                                                                                                                                                          \
          res[m][n_loop] = mmac<scalar_t, float>(vec4_Element<scalar_t>{input_vec8_fp16[m][k_idx].fp16[0],                                                                                \
                                                                        input_vec8_fp16[m][k_idx].fp16[4],                                                                                \
                                                                        input_vec8_fp16[m][k_idx].fp16[1],                                                                                \
                                                                        input_vec8_fp16[m][k_idx].fp16[5]},                                                                               \
                                                 vec4_Element<scalar_t>{k_reg_half2[k_idx].vec2_i16[0][0],                                                                                \
                                                                        k_reg_half2[k_idx].vec2_i16[0][1],                                                                                \
                                                                        k_reg_half2[k_idx].vec2_i16[1][0],                                                                                \
                                                                        k_reg_half2[k_idx].vec2_i16[1][1]},                                                                               \
                                                 res[m][n_loop]);                                                                                                                         \
          res[m][n_loop] = mmac<scalar_t, float>(vec4_Element<scalar_t>{input_vec8_fp16[m][k_idx].fp16[2],                                                                                \
                                                                        input_vec8_fp16[m][k_idx].fp16[6],                                                                                \
                                                                        input_vec8_fp16[m][k_idx].fp16[3],                                                                                \
                                                                        input_vec8_fp16[m][k_idx].fp16[7]},                                                                               \
                                                 vec4_Element<scalar_t>{k_reg_half2[k_idx].vec2_i16[2][0],                                                                                \
                                                                        k_reg_half2[k_idx].vec2_i16[2][1],                                                                                \
                                                                        k_reg_half2[k_idx].vec2_i16[3][0],                                                                                \
                                                                        k_reg_half2[k_idx].vec2_i16[3][1]},                                                                               \
                                                 res[m][n_loop]);                                                                                                                         \
        }                                                                                                                                                                                 \
        asm volatile("s_setprio 0");                                                                                                                                                      \
      }                                                                                                                                                                                   \
      __builtin_amdgcn_sched_barrier(0);                                                                                                                                                  \
                                                                                                                                                                                          \
    } /*stage 0 mmac*/                                                                                                                                                                    \
  } /*stage 0*/                                                                                                                                                                           \
                                                                                                                                                                                          \
  /*========================== ds read -> mma compute  stage 1 ====================================*/                                                                                     \
  }                                                                                                                                                                                       \
  }

#define int4_moe_wna16_cal_out(n_loop)                                                                                                              \
  {                                                                                                                                                 \
      /*======================= output ======================*/                                                                                     \
      {                                                                                                                                             \
          int offset_n = blockIdx.y * BLOCK_SIZE_N_LOOPS * BLOCK_SIZE_N + n_loop * BLOCK_SIZE_N + warp_id * WARP_N + laneid_shfl_4 * (WARP_N / 16); \
                                                                                                                                                    \
  {                                                                                                                                                 \
                                                                                                                                                    \
    _Pragma("unroll") for (int m = 0; m < (BLOCK_SIZE_M + 15) / 16 /*num_valid_tokens*/; ++m)                                                       \
    {                                                                                                                                               \
      int32_t token_index = token_index_arr_new[m];                                                                                                 \
      if (token_index / top_k < size_m)                                                                                                             \
      {                                                                                                                                             \
                                                                                                                                                    \
        /*乘以对应expert的权重*/                                                                                                                    \
        if (mul_topk_weight)                                                                                                                        \
        {                                                                                                                                           \
          for (int vec_id = 0; vec_id < 4; vec_id++)                                                                                                \
          {                                                                                                                                         \
            res[m][n_loop][vec_id] *= topk_weights[token_index];                                                                                    \
          }                                                                                                                                         \
        }                                                                                                                                           \
                                                                                                                                                    \
        /*if(k_loop == (BLOCK_SIZE_K_LOOPS - 1)) */                                                                                                 \
        {                                                                                                                                           \
          if (USE_ATOMIC)                                                                                                                           \
          {                                                                                                                                         \
            for (int vec_id = 0; vec_id < 4; vec_id++)                                                                                              \
            {                                                                                                                                       \
              __builtin_hcu_global_atomic_fadd_f32(&output[token_index * size_n + offset_n + vec_id * 4 * (WARP_N / 16)], res[m][n_loop][vec_id]);  \
            }                                                                                                                                       \
          }                                                                                                                                         \
          else                                                                                                                                      \
          {                                                                                                                                         \
            scalar_t *output_fp16 = reinterpret_cast<scalar_t *>(output);                                                                           \
            for (int vec_id = 0; vec_id < 4; vec_id++)                                                                                              \
            {                                                                                                                                       \
              output_fp16[token_index * size_n + offset_n + vec_id * 4 * (WARP_N / 16)] = Dtype::float2num(res[m][n_loop][vec_id]);                 \
            }                                                                                                                                       \
          }                                                                                                                                         \
        }                                                                                                                                           \
      }                                                                                                                                             \
    }                                                                                                                                               \
  }                                                                                                                                                 \
  }                                                                                                                                                 \
  }

template <typename scalar_t,
          int bit,
          uint32_t top_k,
          uint32_t BLOCK_SIZE_M,
          uint32_t BLOCK_SIZE_N,
          uint32_t BLOCK_SIZE_K,
          bool has_zp,
          bool mul_topk_weight,
          uint32_t group_size,
          int BLOCK_SIZE_M_LOOPS,
          int BLOCK_SIZE_N_LOOPS,
          int BLOCK_SIZE_K_LOOPS,
          bool USE_ATOMIC,
          int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(256, 1) moe_wna16_gemm_kernel(
    const scalar_t *__restrict__ input,
    float *__restrict__ output,
    const uint32_t *__restrict__ qweight,
    const scalar_t *__restrict__ scales,
    const uint32_t *__restrict__ qzeros,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k)
{ // kernel enterpoint
  if (blockIdx.z * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return;

  constexpr int WARP_N = 16;
  constexpr int WARP_NUM = BLOCK_SIZE_N / WARP_N;
  const uint32_t GROUPS = BLOCK_SIZE_K / group_size;

  auto g_input = tcp_cache_swizzle_func<64, scalar_t>(input);
  auto g_sorted_token_ids = tcp_cache_swizzle_func<1, int32_t>(sorted_token_ids);

  int warp_id_vec = (threadIdx.x / 64); // warp id in a block
  int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
  int lane_id = (threadIdx.x & 63); // lane id, 0-63

  int laneid_shfl_4 = lane_id >> 4;
  int laneid_and_15 = lane_id & 15;
  int w_lane_n_idx = laneid_shfl_4; // ((WARP_N/16 == 2) ?((laneid_shfl_4 & 1)*2 + (laneid_shfl_4 >> 1)) : laneid_shfl_4); // (0, 1, 2, 3) --> (0, 2, 1, 3) //laneid_shfl_4

  int w_lane_k_idx = laneid_and_15;

  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;

  int expert_id = expert_ids[blockIdx.z];
  if (expert_ids[blockIdx.z] == -1)
    return;
  int32_t num_valid_tokens[BLOCK_SIZE_M_LOOPS] = {0};
  extern __shared__ uint16_t block_input_tmp[];
  scalar_t *block_input = reinterpret_cast<scalar_t *>(block_input_tmp);
  uint32_t *weight_lds = reinterpret_cast<uint32_t *>(block_input); // lds
  scalar_t2 *block_input_half2 = reinterpret_cast<scalar_t2 *>(block_input);
  union_vec_array<(BLOCK_SIZE_M / 4)> token_index_arr;
  int32_t token_index_arr_new[(BLOCK_SIZE_M + 15) / 16];

  constexpr int K_TILE = 128;
  constexpr int32_t pack_factor = 32 / bit; // 4字节能存多少元素
  constexpr int STAGES = 1;
  scalar_t2 res2;
  scalar_t2 scale_f2[2];
  scalar_t2 qzero_f2[2];
  scalar_t2 scale_f2_0;
  scalar_t2 qzero_f2_0;

  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;     // TODO
  const uint64_t expert_offset_weight = ((uint64_t)size_n) * 128 * expert_id; // TODO

  const uint32_t *expert_qweight = qweight + expert_offset_weight / pack_factor;
  auto g_expert_qweight = tcp_cache_swizzle_func<64, uint32_t>(expert_qweight);

  const scalar_t *expert_scales = scales + expert_offset / group_size;
  const uint32_t *expert_qzeros = qzeros + expert_offset / group_size / pack_factor; // n方向两个int4拼成int8

  auto g_expert_qzero = tcp_cache_swizzle_func<1, uint8_t>(reinterpret_cast<const uint8_t *>(expert_qzeros));

  union_vec4_int8 expert_w_reg[4];

  union_vec<scalar_t, GROUPS> expert_scales_groups;
  union_vec<uint8_t, GROUPS> expert_qzeros_groups;

  union_vec8_fp16<scalar_t> input_vec8_fp16[(BLOCK_SIZE_M + 15) / 16][BLOCK_SIZE_K_LOOPS * (BLOCK_SIZE_K / 32)];

  vec4_fp32 res[(BLOCK_SIZE_M + 15) / 16][4] = {0};
  int lds_stage_offset;
  int k_loop = 0;
  int stage_id;
  uint32_t uint32_zero_tmp[2];

  if (BLOCK_SIZE_N_LOOPS == 1)
  {
    for (int k_loop = 0; k_loop < BLOCK_SIZE_K_LOOPS; k_loop++)
    {
      int4_moe_wna16_cal(0);
    }
    int4_moe_wna16_cal_out(0);
  }
  else if (BLOCK_SIZE_N_LOOPS == 2)
  {
    for (int k_loop = 0; k_loop < BLOCK_SIZE_K_LOOPS; k_loop++)
    {
      int4_moe_wna16_cal(0);
      int4_moe_wna16_cal(1);
    }
    int4_moe_wna16_cal_out(0);
    int4_moe_wna16_cal_out(1);
  }
  else if (BLOCK_SIZE_N_LOOPS == 4)
  {
    for (int k_loop = 0; k_loop < BLOCK_SIZE_K_LOOPS; k_loop++)
    {
      int4_moe_wna16_cal(0);
      int4_moe_wna16_cal(1);
      int4_moe_wna16_cal(2);
      int4_moe_wna16_cal(3);
    }
    int4_moe_wna16_cal_out(0);
    int4_moe_wna16_cal_out(1);
    int4_moe_wna16_cal_out(2);
    int4_moe_wna16_cal_out(3);
  }
  else
  {
    static_assert("NOT SUPPORT");
  }
} // kernel

template <typename scalar_t,
          int bit,
          int top_k,
          int BLOCK_SIZE_M,
          int BLOCK_SIZE_N,
          int BLOCK_SIZE_K,
          bool has_zp,
          bool mul_topk_weight,
          int group_size,
          int BLOCK_SIZE_M_LOOPS,
          int BLOCK_SIZE_N_LOOPS,
          int BLOCK_SIZE_K_LOOPS,
          bool USE_ATOMIC,
          int THREADS_PER_BLOCK>
void run_moe_wna16_gemm(const scalar_t *input,
                        float *output_fp32,
                        const uint32_t *b_qweight,
                        const scalar_t *b_scales,
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

  gridDim.z = num_token_blocks;
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N * BLOCK_SIZE_N_LOOPS);
  gridDim.x = DIVIDE(size_k, BLOCK_SIZE_K * BLOCK_SIZE_K_LOOPS);

  auto kernel = moe_wna16_gemm_kernel<scalar_t, bit, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight, group_size, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC, THREADS_PER_BLOCK>;
  constexpr int K_TILE = 128;
  const int shared_mem_size = 12.5 * 1024;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  kernel<<<gridDim, blockDim, shared_mem_size, stream>>>(
      input,
      output_fp32,
      b_qweight,
      b_scales,
      b_qzeros,
      topk_weights,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_pad,
      size_m, size_n, size_k); //, block_size_k_loops);
}