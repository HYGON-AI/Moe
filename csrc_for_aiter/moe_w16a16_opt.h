#pragma once
#include "moe_w16a16_utils.h"
#include "moe_w16a16_config.h"
namespace at
{
  namespace native
  {

#define BOOL_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    } }()

#define ATOMIC_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    if (COND) {                                 \
      torch::Tensor output_tmp = output.alias();\
      constexpr static bool CONST_NAME = true;  \
      const int BLOCK_SIZE_K_ = 128;            \
      return __VA_ARGS__();                     \
    } else {                                    \
      torch::Tensor output_tmp = output.alias();\
      constexpr static bool CONST_NAME = false; \
      const int BLOCK_SIZE_K_ = 128;            \
      return __VA_ARGS__();                     \
    } }()

    /****************************************kernel*************************************************/

    template <
        typename Element,
        uint16_t WARP_NUM,
        uint16_t BLOCK_SIZE_M,
        uint16_t BLOCK_SIZE_N,
        uint16_t BLOCK_SIZE_K,
        uint16_t WARP_M,
        uint16_t WARP_N,
        uint16_t WARP_K,
        bool mul_topk_weight>
    __attribute__((hcu_co_issue_vgpr_size(256)))
    __global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NN_DECODE_UP(
        const Element *__restrict__ input,
        const Element *__restrict__ qweight,
        Element *__restrict__ output,
        const float *__restrict__ topk_weights,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_ids,
        const int32_t *__restrict__ num_tokens_post_pad,
        uint32_t size_m,
        uint32_t size_n,
        uint32_t size_k,
        uint32_t size_kb,
        uint32_t sorted_token_lens,
        uint32_t top_k,
        uint32_t delta,
        uint32_t expert_num)
    {
      const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
      const int bidy = blockIdx.y; // pid_n
      const int bidz = blockIdx.x; // pid_k

      if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
        return; // 对于无效的block,直接返回
      if (!w16a16_ck_valid_up(sorted_token_ids[bidx * BLOCK_SIZE_M], size_m, top_k))
        return; // 对于无效的block,直接返回
      constexpr int STAGES = 2;
      const uint32_t delta_bidx = bidx / delta;
      const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
      if (expert_id >= expert_num || expert_id < 0)
        return;

      const uint64_t expert_offset = ((uint64_t)size_n) * size_kb * expert_id; // 这个是对应专家的weight偏移

      const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 16; // 具体偏移到对应专家的weight的某一个小的分块

      const uint32_t output_offset = bidy * BLOCK_SIZE_N;

      auto g_qweight = qweight + qweight_offset;
      auto g_input = input;
      Element *g_output;
      g_output = output + output_offset;
      constexpr int mfma_m = 16;
      constexpr int mfma_n = 16;
      constexpr int mfma_k = 16;

      int warp_id_vec = threadIdx.x / 64; // warp id in a block
      int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
      int lane_id = threadIdx.x & 63; // thread_id
      int row_id = lane_id % 16;
      int col_id = lane_id / 16;

      const int warp_n_num = BLOCK_SIZE_N / (WARP_N * 2);
      const int warp_k_num = BLOCK_SIZE_K / WARP_K;

      int warp_k_id = warp_id % warp_k_num;
      int warp_n_id = warp_id / warp_k_num;

      if (bidy * BLOCK_SIZE_N * 16 + warp_n_id * 64 * 8 >= size_n)
        return;
      extern __shared__ uint8_t smem3[]; // 声明lds信息
      float *output_lds = (float *)&(smem3);

      const int *sorted_token_ids_offset = sorted_token_ids;
      f16_vec<WARP_K / 4, Element> A_reg[WARP_M / mfma_m][2];
      f16_vec<WARP_K / 2, Element> B_reg[WARP_N / mfma_n][2];

      floatx4 C_reg[1][2 * (WARP_M / 16) * (WARP_N / 16)];
      for (int init = 0; init < (2 * (WARP_M / 16) * (WARP_N / 16)); init++)
      {
        C_reg[0][init] = {0.0, 0.0, 0.0, 0.0};
      }
      if (WARP_K == 32)
      {
        MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_kpack2_kernel<false, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>(g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy);
      }
      else
      {
        MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_kernel<false, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>(g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy);
      }
      if (warp_k_id == 0)
      {
        using vec_bf16_2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
        for (int m_tile = 0; m_tile < WARP_M / mfma_m; m_tile++)
        {
          int sorted_token_id = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + m_tile * mfma_m + row_id, int(sorted_token_lens - 1))];
          int token_index = w16a16_ck_flat_id(sorted_token_id, top_k);
          for (int i = 0; i < 4; i++)
          {
            if (w16a16_ck_valid_up(sorted_token_id, size_m, top_k))
            {
              vec_bf16_2 D;
              for (int j = 0; j < 2; j++)
              {
                (reinterpret_cast<uint16_t *>(&D))[j] = w16_f32_to_f16<Element>((C_reg[0][m_tile * 2 + j])[i]);
              }
              *reinterpret_cast<vec_bf16_2 *>(&g_output[token_index * (size_n >> 4) + warp_n_id * WARP_N * 2 + col_id * 2 + i * 8]) = D;
            }
          }
        }
      }
    }

    template <
        typename Element,
        uint16_t WARP_NUM,
        uint16_t BLOCK_SIZE_M,
        uint16_t BLOCK_SIZE_N,
        uint16_t BLOCK_SIZE_K,
        uint16_t WARP_M,
        uint16_t WARP_N,
        uint16_t WARP_K,
        bool mul_topk_weight>
    __attribute__((hcu_co_issue_vgpr_size(256)))
    __global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN(
        const Element *__restrict__ input,
        const Element *__restrict__ qweight,
        Element *__restrict__ output,
        const float *__restrict__ topk_weights,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_ids,
        const int32_t *__restrict__ num_tokens_post_pad,
        uint32_t size_m,
        uint32_t size_n,
        uint32_t size_k,
        uint32_t size_kb,
        uint32_t sorted_token_lens,
        uint32_t top_k,
        uint32_t delta,
        uint32_t expert_num)
    {
      const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
      const int bidy = blockIdx.y; // pid_n
      const int bidz = blockIdx.x; // pid_k

      if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
        return; // 对于无效的block,直接返回
      if (!w16a16_ck_valid_down(sorted_token_ids[bidx * BLOCK_SIZE_M], size_m, top_k))
        return; // 对于无效的block,直接返回
      constexpr int STAGES = 2;
      const uint32_t delta_bidx = bidx / delta;
      const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引
      if (expert_id >= expert_num || expert_id < 0)
        return;

      const uint64_t expert_offset = ((uint64_t)size_n) * size_kb * expert_id; // 这个是对应专家的weight偏移

      const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 16; // 具体偏移到对应专家的weight的某一个小的分块

      const uint32_t output_offset = bidy * BLOCK_SIZE_N;

      auto g_qweight = qweight + qweight_offset;
      auto g_input = input;
      Element *g_output;
      g_output = output + output_offset;
      constexpr int mfma_m = 16;
      constexpr int mfma_n = 16;
      constexpr int mfma_k = 16;

      int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
      int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
      int lane_id = threadIdx.x & 63;                            // thread_id
      int row_id = lane_id % 16;
      int col_id = lane_id / 16;

      const int warp_n_num = BLOCK_SIZE_N / (WARP_N * 2);
      const int warp_k_num = BLOCK_SIZE_K / WARP_K;

      int warp_k_id = warp_id % warp_k_num;
      int warp_n_id = warp_id / warp_k_num;
      if (bidy * BLOCK_SIZE_N * 16 + warp_n_id * 64 * 8 >= size_n)
        return;
      extern __shared__ uint8_t smem3[]; // 声明lds信息
      float *output_lds = (float *)&(smem3);

      const int *sorted_token_ids_offset = sorted_token_ids;
      f16_vec<WARP_K / 4, Element> A_reg[WARP_M / mfma_m][2];
      f16_vec<WARP_K / 2, Element> B_reg[WARP_N / mfma_n][2];

      floatx4 C_reg[1][2 * (WARP_M / 16) * (WARP_N / 16)];
      for (int init = 0; init < (2 * (WARP_M / 16) * (WARP_N / 16)); init++)
      {
        C_reg[0][init] = {0.0, 0.0, 0.0, 0.0};
      }
      if (WARP_K == 32)
      {
        MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN_kpack2_kernel<false, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>(g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy, topk_weights);
      }
      else
      {
        MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN_kernel<false, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>(g_input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy, topk_weights);
      }
      if (warp_k_id == 0)
      {
        using vec_bf16_2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) unsigned short;
        for (int m_tile = 0; m_tile < WARP_M / mfma_m; m_tile++)
        {
          int sorted_token_id = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + m_tile * mfma_m + row_id, int(sorted_token_lens - 1))];
          int token_index = w16a16_ck_flat_id(sorted_token_id, top_k);
          if (w16a16_ck_valid_down(sorted_token_id, size_m, top_k))
          {
            float topk_weight_value = (topk_weights + token_index)[0];
            for (int i = 0; i < 4; i++)
            {
              vec_bf16_2 D;
              for (int j = 0; j < 2; j++)
              {
                (reinterpret_cast<uint16_t *>(&D))[j] = w16_f32_to_f16<Element>((C_reg[0][m_tile * 2 + j])[i] * topk_weight_value);
              }
              *reinterpret_cast<vec_bf16_2 *>(&g_output[token_index * (size_n >> 4) + warp_n_id * WARP_N * 2 + col_id * 2 + i * 8]) = D;
            }
          }
        }
      }
    }

    template <
        typename Element,
        uint16_t WARP_NUM,
        uint16_t BLOCK_SIZE_M,
        uint16_t BLOCK_SIZE_N,
        uint16_t BLOCK_SIZE_K,
        uint16_t WARP_M,
        uint16_t WARP_N,
        uint16_t WARP_K,
        bool mul_topk_weight,
        bool tail_N_process>
    __attribute__((hcu_co_issue_vgpr_size(256)))
    __global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP(
        const Element *__restrict__ input,
        const Element *__restrict__ qweight,
        Element *__restrict__ output,
        const float *__restrict__ topk_weights,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_ids,
        const int32_t *__restrict__ num_tokens_post_pad,
        uint32_t size_m,
        uint32_t size_n,
        uint32_t size_k,
        uint32_t size_kb,
        uint32_t sorted_token_lens,
        uint32_t top_k,
        uint32_t delta,
        uint32_t expert_num)
    {
      const int bidx = blockIdx.z; // m方向，分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
      const int bidy = blockIdx.y; // n方向
      const int bidz = blockIdx.x;
      const int bytes_per_element = 2;
      const int marlin_warp_k = 16; // marlin weight tile: 16x32
      const int marlin_warp_n = 32;
      const int marlin_warp_size = marlin_warp_k * marlin_warp_n;
      const int marlin_stride_b = size_n * bytes_per_element;
      const int real_size_n = (size_n * size_kb) / size_k;

      if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
        return; // 对于无效的block,直接返回
      if (!w16a16_ck_valid_up(sorted_token_ids[bidx * BLOCK_SIZE_M], size_m, top_k))
        return; // 对于无效的block,直接返回

      constexpr int STAGES = BLOCK_SIZE_K / WARP_K; // 1
      const uint32_t delta_bidx = bidx / delta;
      const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

      if (expert_id >= expert_num || expert_id < 0)
        return;

      const uint64_t expert_offset = ((uint64_t)size_n) * size_kb * expert_id;                                  // 这个是对应专家的weight偏移
      const uint64_t qweight_offset = expert_offset + bidy * (BLOCK_SIZE_N / marlin_warp_n) * marlin_warp_size; // 具体偏移到对应专家的weight的某一个小的分块

      const uint32_t output_offset = bidy * BLOCK_SIZE_N;
      auto g_qweight = qweight + qweight_offset;
      auto g_input = input;

      Element *g_output;
      extern __shared__ uint8_t smem[];                                    // 声明lds信息
      Element *input_lds = (Element *)&(smem);                             // 输入lds首地址,首地址开始分配给input
      Element *qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
      uint16_t *output_lds = (uint16_t *)&(smem);                          // 重复使用lds,留给output_lds

      g_output = output + output_offset;

      constexpr int mfma_m = 16;
      constexpr int mfma_n = 16;
      constexpr int mfma_k = 16;

      int warp_id_vec = threadIdx.x / 64;
      int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
      int lane_id = threadIdx.x & 63;
      int row_id = lane_id % 16;
      int col_id = lane_id / 16;

      const int warp_n_num = BLOCK_SIZE_N / WARP_N;

      int warp_n_id = warp_id % warp_n_num; // 0
      int warp_m_id = warp_id / warp_n_num; // 0
      constexpr int dwordx4_num_elements = 8;
      constexpr int N_thread = BLOCK_SIZE_N / dwordx4_num_elements; // N方向需要的线程数 使用dwordx4即8个bf16
      const int m_num_threads = 64 / N_thread;                      // 要求BLOCK_SIZE_N <= 512
      const int n_num_threads = 64 / m_num_threads;
      const int m_thread_idx = lane_id / n_num_threads;
      const int n_thread_idx = lane_id & (n_num_threads - 1);

      const int *sorted_token_ids_offset = sorted_token_ids;

      w16_union_vec<Element, 4> A_reg[WARP_M / mfma_m * STAGES][WARP_K / mfma_k];
      w16_union_vec<Element, 8> B_reg[WARP_N / marlin_warp_n][WARP_K / marlin_warp_k];

      floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};
      int token_index_arr[BLOCK_SIZE_M / (m_num_threads * WARP_NUM)];
#pragma unroll
      for (int m_idx = 0; m_idx < BLOCK_SIZE_M / (m_num_threads * WARP_NUM); m_idx++)
      {
        const int token_index = w16a16_ck_flat_id(sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx * m_num_threads * WARP_NUM + warp_id * m_num_threads + m_thread_idx, int(sorted_token_lens - 1))], top_k);
        token_index_arr[m_idx] = token_index;
      }

      if (tail_N_process)
      {
        MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_remainder<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>(g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy);
      }
      else
      {
        MOE_W16A16_MARLIN_HIP_NT_prefill_kernel<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element>(g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
      }

#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / marlin_warp_n; min_tile_n++)
        {
#pragma unroll
          for (int n_tile = 0; n_tile < marlin_warp_n / mfma_n; n_tile++)
          {
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {
              float value = C_reg[0][min_tile_m * ((WARP_N / marlin_warp_n) * (marlin_warp_n / mfma_n)) + min_tile_n * (marlin_warp_n / mfma_n) + n_tile][reg_id];
              int index = warp_m_id * WARP_M * BLOCK_SIZE_N + min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * marlin_warp_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N +
                          reg_id * (marlin_warp_n / mfma_n) * 4 + (lane_id / 16) * (marlin_warp_n / mfma_n) + n_tile + (min_tile_m * mfma_m + warp_m_id * WARP_M + (lane_id % 16)) / 2 * 2 /*padding*/; // + (lane_id & 15 )/2 * 2/*padding*/

              output_lds[index] = w16_f32_to_f16<Element>(value);
            }
          }
        }
      }

      __syncthreads();

      {

        using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
#pragma unroll
        for (int m_idx = 0; m_idx < BLOCK_SIZE_M; m_idx += m_num_threads * WARP_NUM)
        {
          const int32_t token_index = token_index_arr[m_idx / (m_num_threads * WARP_NUM)];
#pragma unroll
          for (int n_idx = 0; n_idx < BLOCK_SIZE_N; n_idx += n_num_threads * dwordx4_num_elements)
          {
            if ((token_index < size_m * top_k) && (bidy * BLOCK_SIZE_N + n_idx + n_thread_idx * dwordx4_num_elements < (size_n >> 4)))
            {
              *reinterpret_cast<vec_bf16_8 *>(&g_output[token_index * real_size_n + n_idx + n_thread_idx * dwordx4_num_elements]) =
                  *reinterpret_cast<vec_bf16_8 *>(&output_lds[(m_idx + m_thread_idx + warp_id * m_num_threads) * BLOCK_SIZE_N + n_idx + n_thread_idx * dwordx4_num_elements + (m_idx + m_thread_idx + warp_id * m_num_threads) / 2 * 2 /*padding*/]);
            }
          }
        }
      }
    }

    template <
        typename Element,
        uint16_t WARP_NUM,
        uint16_t BLOCK_SIZE_M,
        uint16_t BLOCK_SIZE_N,
        uint16_t BLOCK_SIZE_K,
        uint16_t WARP_M,
        uint16_t WARP_N,
        uint16_t WARP_K,
        bool mul_topk_weight,
        bool tail_k_process> // true
    __attribute__((hcu_co_issue_vgpr_size(256)))
    __global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN(
        const Element *__restrict__ input,
        const Element *__restrict__ qweight,
        Element *__restrict__ output,
        const float *__restrict__ topk_weights,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_ids,
        const int32_t *__restrict__ num_tokens_post_pad,
        uint32_t size_m,
        uint32_t size_n,
        uint32_t size_k,
        uint32_t size_kb,
        uint32_t sorted_token_lens,
        uint32_t top_k,
        uint32_t delta)
    {
      const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
      const int bidy = blockIdx.y; // pid_n
      const int bidz = blockIdx.x; // pid_k
      const int bytes_per_element = 2;
      const int marlin_warp_k = 16; // marlin weight tile: 16x32
      const int marlin_warp_n = 32;
      const int marlin_warp_size = marlin_warp_k * marlin_warp_n;
      const int marlin_stride_b = size_n * bytes_per_element;
      const int real_size_n = (size_n * size_kb) / size_k;

      if (bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0] || !w16a16_ck_valid_down(sorted_token_ids[bidx * BLOCK_SIZE_M], size_m, top_k))
        return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

      const int STAGES = BLOCK_SIZE_K / WARP_K; // 采用两stage
      const uint32_t delta_bidx = bidx / delta;
      const int32_t expert_id = expert_ids[delta_bidx];                                                         // 专家的索引
      const uint64_t expert_offset = ((uint64_t)size_n) * size_kb * expert_id;                                  // 这个是对应专家的weight偏移
      const uint64_t qweight_offset = expert_offset + bidy * (BLOCK_SIZE_N / marlin_warp_n) * marlin_warp_size; // 具体偏移到对应专家的weight的某一个小的分块
      const uint32_t output_offset = bidy * BLOCK_SIZE_N;                                                       // 计算之后是mxn,这应该是计算输出n方向的位置

      auto g_input = input;
      Element *g_output;
      g_output = output + output_offset;

      constexpr int mfma_m = 16;
      constexpr int mfma_n = 16;
      constexpr int mfma_k = 16;

      int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
      int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
      int lane_id = threadIdx.x & 63;                            // thread_id
      int row_id = lane_id % 16;
      int col_id = lane_id / 16;
      const int warp_n_num = BLOCK_SIZE_N / WARP_N;
      const int warp_m_num = WARP_NUM / warp_n_num;
      const int warps_m_once = warp_m_num * mfma_m;
      int warp_n_id = warp_id % warp_n_num;
      int warp_m_id = warp_id / warp_n_num;
      constexpr int dwordx4_num_elements = 8;
      constexpr int N_thread = BLOCK_SIZE_N / dwordx4_num_elements; // N方向需要的线程数 使用dwordx4即8个bf16
      const int m_num_threads = 64 / N_thread;                      // 要求BLOCK_SIZE_N <= 512
      const int n_num_threads = 64 / m_num_threads;
      const int m_thread_idx = lane_id / n_num_threads;
      const int n_thread_idx = lane_id & (n_num_threads - 1);
      const int write_out_m_size_once = m_num_threads * WARP_NUM; // layout: warp_numx1
      const int stride_in_mfma = mfma_m / write_out_m_size_once;

      extern __shared__ uint8_t smem[];                                    // 声明lds信息
      Element *input_lds = (Element *)&(smem);                             // 输入lds首地址,首地址开始分配给input
      Element *qweight_lds = input_lds + STAGES * (BLOCK_SIZE_M * WARP_K); // 按照最简单的方式分块
      uint16_t *output_lds = (uint16_t *)&(smem);                          // 重复使用lds,留给output_lds

      const int *sorted_token_ids_offset = sorted_token_ids;

      w16_union_vec<Element, 4> A_reg[WARP_M / mfma_m * STAGES][WARP_K / mfma_k]; //
      w16_union_vec<Element, 8> B_reg[WARP_N / marlin_warp_n][WARP_K / marlin_warp_k];

      auto g_qweight = qweight + qweight_offset;

      floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0};

      float weight_value_arr[WARP_M / mfma_m];
      int token_index_arr[BLOCK_SIZE_M / write_out_m_size_once];

      if (stride_in_mfma == 0)
      {
#pragma unroll // warp_m_num > 1 eg: warp_m_num = 2 m方向的warp一次可以产生2个mfma_m的数据，并且间隔WARP_M
        for (int m_idx = 0; m_idx < BLOCK_SIZE_M / write_out_m_size_once; m_idx++)
        {
          const int block_m_idx = m_idx * mfma_m + ((warp_id * m_num_threads + m_thread_idx) / mfma_m) * WARP_M + ((warp_id * m_num_threads + m_thread_idx) % mfma_m);
          const int token_index = w16a16_ck_flat_id(sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + block_m_idx, int(sorted_token_lens - 1))], top_k);
          token_index_arr[m_idx] = token_index;
        }
      }
      else
      {
#pragma unroll // warp_m_num == 1
        for (int m_idx = 0; m_idx < BLOCK_SIZE_M / mfma_m; m_idx++)
        {
#pragma unroll
          for (int m_mfma_idx = 0; m_mfma_idx < mfma_m / write_out_m_size_once; m_mfma_idx++)
          {
            const int block_m_idx = m_idx * mfma_m + m_mfma_idx * write_out_m_size_once + ((warp_id * m_num_threads + m_thread_idx) / mfma_m) * WARP_M + ((warp_id * m_num_threads + m_thread_idx) % mfma_m);
            const int token_index = w16a16_ck_flat_id(sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + block_m_idx, int(sorted_token_lens - 1))], top_k);
            token_index_arr[m_idx * (mfma_m / write_out_m_size_once) + m_mfma_idx] = token_index;
          }
        }
      }

#pragma unroll
      for (int idx = 0; idx < WARP_M / mfma_m; idx++)
      {
        int sorted_token_id = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + warp_m_id * WARP_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
        int token_index = w16a16_ck_flat_id(sorted_token_id, top_k);
        int token_index_safe = std::min(token_index, int(size_m * top_k - 1));
        weight_value_arr[idx] = w16a16_ck_valid_down(sorted_token_id, size_m, top_k) ? topk_weights[token_index_safe] : 0.0f;
      }

      if (BLOCK_SIZE_K == WARP_K)
        MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_DOWN_Blocksizek32_weight_bypass_lds<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, tail_k_process, Element>(g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy);
      else
      {
        MOE_W16A16_MARLIN_HIP_NT_prefill_kernel_DOWN<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, tail_k_process, Element>(g_input, g_qweight, input_lds, qweight_lds, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_n, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy);
      }
#pragma unroll
      for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
      {
        float weight_value = weight_value_arr[min_tile_m];
#pragma unroll
        for (int min_tile_n = 0; min_tile_n < WARP_N / marlin_warp_n; min_tile_n++)
        {
#pragma unroll
          for (int n_tile = 0; n_tile < marlin_warp_n / mfma_n; n_tile++)
          {
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {
              float value = C_reg[0][min_tile_m * ((WARP_N / marlin_warp_n) * (marlin_warp_n / mfma_n)) + min_tile_n * (marlin_warp_n / mfma_n) + n_tile][reg_id] * weight_value;
              int index = warp_m_id * mfma_m * BLOCK_SIZE_N + min_tile_n * marlin_warp_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N +
                          reg_id * (marlin_warp_n / mfma_n) * 4 + (lane_id / 16) * (marlin_warp_n / mfma_n) + n_tile + (warp_m_id * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/; // + (lane_id & 15 )/2 * 2/*padding*/
              output_lds[index] = w16_f32_to_f16<Element>(value);
            }
          }
        }
        __syncthreads();
        using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;
#pragma unroll
        for (int m_mfma_idx = 0; m_mfma_idx < warps_m_once; m_mfma_idx += write_out_m_size_once)
        {
          const int32_t token_index = token_index_arr[min_tile_m * (warps_m_once / write_out_m_size_once) + (m_mfma_idx / write_out_m_size_once)];
#pragma unroll
          for (int n_idx = 0; n_idx < BLOCK_SIZE_N; n_idx += n_num_threads * dwordx4_num_elements)
          {
            if (token_index < size_m)
            {
              *reinterpret_cast<vec_bf16_8 *>(&g_output[token_index * real_size_n + n_idx + n_thread_idx * dwordx4_num_elements]) =
                  *reinterpret_cast<vec_bf16_8 *>(&output_lds[(m_mfma_idx + m_thread_idx + warp_id * m_num_threads) * BLOCK_SIZE_N + n_idx + n_thread_idx * dwordx4_num_elements + (m_mfma_idx + m_thread_idx + warp_id * m_num_threads) / 2 * 2 /*padding*/]);
            }
          }
        }
        lgkmcnt_wait_barrier(0);
      }
    }

    /****************************************marlin layout launch*************************************************/
    template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
    void launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN(const GemmParams3<T> &params)
    {

      constexpr int kwarp = BLOCK_SIZE_K / WARP_K;
      constexpr int nwarp = BLOCK_SIZE_N / (WARP_N * 2);
      constexpr int WARP_NUM = kwarp * nwarp;
      const bool mul_topk_weight = true;
      dim3 blockDim, gridDim;
      blockDim.x = WARP_NUM * 64;
      blockDim.y = 1;
      blockDim.z = 1;

      gridDim.z = std::min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), params.size_m * params.delta); // m方向
      gridDim.y = DIVIDE(params.size_n, 16 * BLOCK_SIZE_N);                                               // n方向
      gridDim.x = 1;                                                                                      // k方向

      int lds_size = (kwarp - 1) * nwarp * 36 * 16 * 4;
      const hipStream_t stream = at::cuda::getCurrentHIPStream();

      MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                           mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
          params.ptr_A,
          params.ptr_B0,
          params.ptr_C,
          params.topk_weights,
          params.sorted_token_ids,
          params.expert_ids,
          params.num_tokens_post_pad_ptr,
          params.size_m,
          params.size_n,
          params.size_k,
          params.size_kofb,
          params.sorted_token_lens,
          params.top_k,
          params.delta,
          params.expert_num);

      auto err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
    }

    template <
        typename Element,
        uint16_t WARP_NUM,
        uint16_t BLOCK_SIZE_M,
        uint16_t BLOCK_SIZE_N,
        uint16_t BLOCK_SIZE_K,
        uint16_t WARP_M,
        uint16_t WARP_N,
        uint16_t WARP_K,
        uint16_t N_LOOP,
        uint16_t STATIC_K,
        uint16_t STATIC_N,
        uint16_t STATIC_OUTPUT_N,
        uint16_t STATIC_LOOP_K,
        uint16_t CACHE_MODE,
        bool FUSE_NTILE2,
        bool mul_topk_weight>
    __global__ void __launch_bounds__(1024) MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP(
        const Element *__restrict__ input,
        const Element *__restrict__ qweight,
        Element *__restrict__ output,
        const float *__restrict__ topk_weights,
        const int32_t *__restrict__ sorted_token_ids,
        const int32_t *__restrict__ expert_ids,
        const int32_t *__restrict__ num_tokens_post_pad,
        uint32_t size_m,
        uint32_t size_n,
        uint32_t size_k,
        uint32_t size_kb,
        uint32_t sorted_token_lens,
        uint32_t top_k,
        uint32_t delta,
        uint32_t expert_num)
    {
      const int bidx = blockIdx.z;
      const int bidy_base = blockIdx.y;
      if (top_k != 1 || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
        return;

      constexpr int STAGES = 2;
      const uint32_t delta_bidx = bidx / delta;
      const int32_t expert_id = expert_ids[delta_bidx];
      if (expert_id >= expert_num || expert_id < 0)
        return;

      constexpr int mfma_m = 16;
      constexpr int mfma_n = 16;
      int warp_id_vec = threadIdx.x / 64;
      int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
      int lane_id = threadIdx.x & 63;
      int row_id = lane_id % 16;
      int col_id = lane_id / 16;

      constexpr int warp_k_num = BLOCK_SIZE_K / WARP_K;
      const int warp_k_id = warp_id % warp_k_num;
      const int warp_n_id = warp_id / warp_k_num;
      constexpr uint32_t static_weight_stride_n = STATIC_N > 0 ? STATIC_N : 0;
      constexpr uint32_t static_output_n = STATIC_OUTPUT_N > 0 ? STATIC_OUTPUT_N : STATIC_N;
      const uint32_t weight_stride_n = static_weight_stride_n > 0 ? static_weight_stride_n : size_n;
      const uint32_t output_logical_n = static_output_n > 0 ? static_output_n : size_n;
      const uint32_t output_stride_n = output_logical_n >> 4;
      const int *sorted_token_ids_offset = sorted_token_ids;
      const uint64_t expert_offset = ((uint64_t)weight_stride_n) * size_kb * expert_id;

      if constexpr (FUSE_NTILE2)
      {
        const int bidy0 = bidy_base * N_LOOP;
        if (bidy0 * BLOCK_SIZE_N * 16 + warp_n_id * 64 * 8 < output_logical_n)
        {
          const bool second_n_valid = (bidy0 + 1) * BLOCK_SIZE_N * 16 + warp_n_id * 64 * 8 < output_logical_n;
          const uint64_t qweight_offset = expert_offset + bidy0 * BLOCK_SIZE_N * 16;
          const uint32_t output_offset0 = bidy0 * BLOCK_SIZE_N;
          const uint32_t output_offset1 = (bidy0 + 1) * BLOCK_SIZE_N;
          const Element *g_qweight = qweight + qweight_offset;
          Element *g_output0 = output + output_offset0;
          Element *g_output1 = output + output_offset1;

          f16_vec<WARP_K / 4, Element> A_reg[WARP_M / mfma_m][2];
          f16_vec<WARP_K / 2, Element> B0_reg[WARP_N / mfma_n][2];
          f16_vec<WARP_K / 2, Element> B1_reg[WARP_N / mfma_n][2];

          floatx4 C0_reg[1][2 * (WARP_M / 16) * (WARP_N / 16)];
          floatx4 C1_reg[1][2 * (WARP_M / 16) * (WARP_N / 16)];
          for (int init = 0; init < (2 * (WARP_M / 16) * (WARP_N / 16)); init++)
          {
            C0_reg[0][init] = {0.0, 0.0, 0.0, 0.0};
            C1_reg[0][init] = {0.0, 0.0, 0.0, 0.0};
          }

          MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_kpack2_ntile2_kernel<WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, Element, STATIC_K, STATIC_N, STATIC_LOOP_K, CACHE_MODE>(input, g_qweight, size_m, A_reg, B0_reg, B1_reg, C0_reg, C1_reg, warp_id, size_k, weight_stride_n, sorted_token_ids_offset, sorted_token_lens, bidx, second_n_valid);

          if (warp_k_id == 0)
          {
            using vec_bf16_2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
            using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
            for (int m_tile = 0; m_tile < WARP_M / mfma_m; m_tile++)
            {
              int sorted_token_id = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + m_tile * mfma_m + row_id, int(sorted_token_lens - 1))];
              int token_index = w16a16_ck_token_id(sorted_token_id);
              if (w16a16_ck_valid_up(sorted_token_id, size_m, 1))
              {
                for (int i = 0; i < 4; i++)
                {
                  vec_bf16_2 D0;
                  vec_bf16_2 D1;
#pragma unroll
                  for (int j = 0; j < 2; j++)
                  {
                    (reinterpret_cast<uint16_t *>(&D0))[j] = w16_f32_to_f16<Element>((C0_reg[0][m_tile * 2 + j])[i]);
                    (reinterpret_cast<uint16_t *>(&D1))[j] = w16_f32_to_f16<Element>((C1_reg[0][m_tile * 2 + j])[i]);
                  }
                  const int gather_base_lane = row_id;
                  uint32_t packed0 = *reinterpret_cast<uint32_t *>(&D0);
                  vec_bf16_8 D0x4;
                  reinterpret_cast<uint32_t *>(&D0x4)[0] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 0 * 16) * 4, packed0);
                  reinterpret_cast<uint32_t *>(&D0x4)[1] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 1 * 16) * 4, packed0);
                  reinterpret_cast<uint32_t *>(&D0x4)[2] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 2 * 16) * 4, packed0);
                  reinterpret_cast<uint32_t *>(&D0x4)[3] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 3 * 16) * 4, packed0);
                  if (col_id == 0)
                  {
                    *reinterpret_cast<vec_bf16_8 *>(&g_output0[token_index * output_stride_n + warp_n_id * WARP_N * 2 + i * 8]) = D0x4;
                  }
                  if (second_n_valid)
                  {
                    uint32_t packed1 = *reinterpret_cast<uint32_t *>(&D1);
                    vec_bf16_8 D1x4;
                    reinterpret_cast<uint32_t *>(&D1x4)[0] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 0 * 16) * 4, packed1);
                    reinterpret_cast<uint32_t *>(&D1x4)[1] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 1 * 16) * 4, packed1);
                    reinterpret_cast<uint32_t *>(&D1x4)[2] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 2 * 16) * 4, packed1);
                    reinterpret_cast<uint32_t *>(&D1x4)[3] = __builtin_amdgcn_ds_bpermute((gather_base_lane + 3 * 16) * 4, packed1);
                    if (col_id == 0)
                    {
                      *reinterpret_cast<vec_bf16_8 *>(&g_output1[token_index * output_stride_n + warp_n_id * WARP_N * 2 + i * 8]) = D1x4;
                    }
                  }
                }
              }
            }
          }
        }
        return;
      }

#pragma unroll
      for (int n_loop = 0; n_loop < N_LOOP; n_loop++)
      {
        const int bidy = bidy_base * N_LOOP + n_loop;
        if (bidy * BLOCK_SIZE_N * 16 + warp_n_id * 64 * 8 >= output_logical_n)
          continue;

        const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 16;
        const uint32_t output_offset = bidy * BLOCK_SIZE_N;
        const Element *g_qweight = qweight + qweight_offset;
        Element *g_output = output + output_offset;

        f16_vec<WARP_K / 4, Element> A_reg[WARP_M / mfma_m][2];
        f16_vec<WARP_K / 2, Element> B_reg[WARP_N / mfma_n][2];

        floatx4 C_reg[1][2 * (WARP_M / 16) * (WARP_N / 16)];
        for (int init = 0; init < (2 * (WARP_M / 16) * (WARP_N / 16)); init++)
        {
          C_reg[0][init] = {0.0, 0.0, 0.0, 0.0};
        }

        MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_kpack2_kernel<false, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, Element, true, STATIC_K, STATIC_N, STATIC_LOOP_K, CACHE_MODE>(input, g_qweight, size_m, A_reg, B_reg, C_reg, warp_id, size_k, weight_stride_n, 1, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx, bidy);

        if (warp_k_id == 0)
        {
          using vec_bf16_2 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
          for (int m_tile = 0; m_tile < WARP_M / mfma_m; m_tile++)
          {
            int sorted_token_id = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + m_tile * mfma_m + row_id, int(sorted_token_lens - 1))];
            int token_index = w16a16_ck_token_id(sorted_token_id);
            if (w16a16_ck_valid_up(sorted_token_id, size_m, 1))
            {
              for (int i = 0; i < 4; i++)
              {
                vec_bf16_2 D;
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                  (reinterpret_cast<uint16_t *>(&D))[j] = w16_f32_to_f16<Element>((C_reg[0][m_tile * 2 + j])[i]);
                }
                *reinterpret_cast<vec_bf16_2 *>(&g_output[token_index * output_stride_n + warp_n_id * WARP_N * 2 + col_id * 2 + i * 8]) = D;
              }
            }
          }
        }
      }
    }

    template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, int N_LOOP, int STATIC_K, int STATIC_N, int STATIC_OUTPUT_N, int STATIC_LOOP_K, int CACHE_MODE, bool FUSE_NTILE2, typename T>
    void launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP(const GemmParams3<T> &params)
    {
      constexpr int kwarp = BLOCK_SIZE_K / WARP_K;
      constexpr int nwarp = BLOCK_SIZE_N / (WARP_N * 2);
      constexpr int WARP_NUM = kwarp * nwarp;
      const bool mul_topk_weight = false;

      dim3 blockDim, gridDim;
      blockDim.x = WARP_NUM * 64;
      blockDim.y = 1;
      blockDim.z = 1;

      const uint32_t m_blocks = std::min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), params.size_m);
      constexpr uint32_t static_launch_n = STATIC_OUTPUT_N > 0 ? STATIC_OUTPUT_N : STATIC_N;
      const uint32_t launch_n = static_launch_n > 0 ? static_launch_n : params.size_n;
      const uint32_t n_blocks = DIV_ceil(launch_n, 16 * BLOCK_SIZE_N * N_LOOP);
      const uint32_t k_blocks = splitk;
      gridDim.x = k_blocks;
      gridDim.y = n_blocks;
      gridDim.z = m_blocks;

      constexpr int lds_size = 24576;
      const hipStream_t stream = at::cuda::getCurrentHIPStream();
      MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                     N_LOOP, STATIC_K, STATIC_N, STATIC_OUTPUT_N, STATIC_LOOP_K, CACHE_MODE, FUSE_NTILE2, mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
          params.ptr_A,
          params.ptr_B0,
          params.ptr_C,
          params.topk_weights,
          params.sorted_token_ids,
          params.expert_ids,
          params.num_tokens_post_pad_ptr,
          params.size_m,
          params.size_n,
          params.size_k,
          params.size_kofb,
          params.sorted_token_lens,
          params.top_k,
          params.delta,
          params.expert_num);
      auto err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        printf("CUDA error in gemm1 topk1 nloop: %s\n", cudaGetErrorString(err));
      }
    }

    template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
    void launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP(const GemmParams3<T> &params)
    {
      constexpr int kwarp = BLOCK_SIZE_K / WARP_K;       // 1
      constexpr int nwarp = BLOCK_SIZE_N / (WARP_N * 2); // 1
      constexpr int WARP_NUM = kwarp * nwarp;            // 1
      const bool mul_topk_weight = false;

      dim3 blockDim, gridDim;
      blockDim.x = WARP_NUM * 64;
      blockDim.y = 1;
      blockDim.z = 1;

      gridDim.z = std::min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), params.size_m * params.top_k); // m方向
      gridDim.y = DIV_ceil(params.size_n, 16 * BLOCK_SIZE_N);                                             // n方向
      gridDim.x = splitk;                                                                                 // k方向

      constexpr int lds_size = 24576;
      const hipStream_t stream = at::cuda::getCurrentHIPStream();
      MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                         mul_topk_weight><<<gridDim, blockDim, lds_size, stream>>>(
          params.ptr_A,
          params.ptr_B0,
          params.ptr_C,
          params.topk_weights,
          params.sorted_token_ids,
          params.expert_ids,
          params.num_tokens_post_pad_ptr,
          params.size_m,
          params.size_n,
          params.size_k,
          params.size_kofb,
          params.sorted_token_lens,
          params.top_k,
          params.delta,
          params.expert_num);
      auto err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
    }
    /****************************************nt layout launch*************************************************/
    template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
    void launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP(const GemmParams3<T> &params)
    {

      constexpr int STAGES = BLOCK_SIZE_K / WARP_K; // 1
      constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
      const bool mul_topk_weight = false;

      dim3 blockDim, gridDim;
      blockDim.x = WARP_NUM * 64;
      blockDim.y = 1;
      blockDim.z = 1;
      constexpr int bytes_per_element = 2;
      const int real_size_n = (params.size_n * params.size_kofb) / params.size_k;
      gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
      gridDim.y = DIV_ceil(real_size_n, BLOCK_SIZE_N);            // n方向 16大小分块 320/128数量为2余64
      gridDim.x = 1;                                              // k方向

      const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K) * bytes_per_element, (BLOCK_SIZE_M * BLOCK_SIZE_N * bytes_per_element + BLOCK_SIZE_M * (BLOCK_SIZE_M / 2) * 2 * bytes_per_element) /*padding*/);
      const hipStream_t stream = at::cuda::getCurrentHIPStream();

      BOOL_SWITCH(((real_size_n % BLOCK_SIZE_N) != 0), tail_N_process, [&]
                  { MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                        mul_topk_weight, tail_N_process><<<gridDim, blockDim, lds_size, stream>>>(
                        params.ptr_A,
                        params.ptr_B0,
                        params.ptr_C,
                        params.topk_weights,
                        params.sorted_token_ids,
                        params.expert_ids,
                        params.num_tokens_post_pad_ptr,
                        params.size_m,
                        params.size_n,
                        params.size_k,
                        params.size_kofb,
                        params.sorted_token_lens,
                        params.top_k,
                        params.delta,
                        params.expert_num); });
      auto err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
    }

    template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
    void launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN(const GemmParams3<T> &params)
    {

      constexpr int STAGES = BLOCK_SIZE_K / WARP_K;
      constexpr int WARP_NUM = (BLOCK_SIZE_M / WARP_M) * (BLOCK_SIZE_N / WARP_N);
      const bool mul_topk_weight = true;

      dim3 blockDim, gridDim;
      blockDim.x = WARP_NUM * 64;
      blockDim.y = 1;
      blockDim.z = 1;
      constexpr int bytes_per_element = 2;
      const int real_size_n = (params.size_n * params.size_kofb) / params.size_k;
      const int mfma_m = 16;
      const int warps_m_once = (BLOCK_SIZE_M / WARP_M) * mfma_m;

      gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
      gridDim.y = DIVIDE(real_size_n, BLOCK_SIZE_N);              // n方向 16大小分块 数量为48
      gridDim.x = 1;                                              // k方向
      const int lds_size = std::max(STAGES * (BLOCK_SIZE_M * WARP_K + BLOCK_SIZE_N * WARP_K * (STAGES - 1)) * bytes_per_element, (warps_m_once * BLOCK_SIZE_N * bytes_per_element + warps_m_once * (warps_m_once / 2) * 2 * bytes_per_element) /*padding*/);
      const hipStream_t stream = at::cuda::getCurrentHIPStream();

      BOOL_SWITCH(((params.size_k % WARP_K) != 0), tail_k_process, [&]
                  { MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                          mul_topk_weight, tail_k_process><<<gridDim, blockDim, lds_size, stream>>>(
                        params.ptr_A,
                        params.ptr_B0,
                        params.ptr_C,
                        params.topk_weights,
                        params.sorted_token_ids,
                        params.expert_ids,
                        params.num_tokens_post_pad_ptr,
                        params.size_m,
                        params.size_n,
                        params.size_k,
                        params.size_kofb,
                        params.sorted_token_lens,
                        params.top_k,
                        params.delta); });

      auto err = cudaGetLastError();
      if (err != cudaSuccess)
      {
        printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
      }
    }

  } // namespace native
} // namespace at
