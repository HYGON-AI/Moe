

#include "moe_w8a16_chan_utils.h"
#include "moe_w8a16_chan_config.h"

template <
    typename Element,
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M,
    uint16_t BLOCK_SIZE_N,
    uint16_t BLOCK_SIZE_K,
    uint16_t WARP_M,
    uint16_t WARP_N,
    uint16_t WARP_K,
    uint16_t GROUP_N,
    uint16_t GROUP_K,
    int STAGES,
    bool mul_topk_weight>
__global__ void __launch_bounds__(256, 1) MOE_W8A16_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP(
    const Element *input,
    const uint32_t *__restrict__ qweight,
    Element *__restrict__ output,
    Element *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t delta)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.x; // pid_n
  const int bidz = blockIdx.y; // pid_k

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  using Dtype = ScalarType<Element>;
  using scalar_t2 = typename ScalarType<Element>::scalar_t2;
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];                              // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;        // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 128;     // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;                            // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = size_n * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块

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
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ uint16_t output_lds[]; // 声明lds信息
  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec_opt_w8a16_A<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt_w8a16<uint32_t, WARP_K / 16> B_int_reg[WARP_N / mfma_n][STAGES];
  Element B_scale_reg[WARP_N / mfma_n][STAGES];
  reg_bf16_fp16<scalar_t2> B_reg[WARP_N / mfma_n][STAGES][WARP_K / 32];

  auto g_qweight = qweight + qweight_offset / 4; // modify

  auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
  auto *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  Element b_scale[(WARP_N / mfma_n) * 4];

  vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64 * (WARP_N / mfma_n), Element>(b_scale_ptr);
#pragma unroll
  for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
  {

    int base_off = min_tile_n * mfma_n;
    b_scale[min_tile_n * 4 + 0] = b_scale_ptr[base_off + col_id + 0];
    b_scale[min_tile_n * 4 + 1] = b_scale_ptr[base_off + col_id + 4];
    b_scale[min_tile_n * 4 + 2] = b_scale_ptr[base_off + col_id + 8];
    b_scale[min_tile_n * 4 + 3] = b_scale_ptr[base_off + col_id + 12];
  }

  floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2

  {
    if (size_k == 7168)
    {
      constexpr static int SIZE_K = 7168;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 6144)
    {
      constexpr static int SIZE_K = 6144;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 5120)
    {
      constexpr static int SIZE_K = 5120;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 4096)
    {
      constexpr static int SIZE_K = 4096;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 3072)
    {
      constexpr static int SIZE_K = 3072;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 2048)
    {
      constexpr static int SIZE_K = 2048;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 1536)
    {
      constexpr static int SIZE_K = 1536;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 1024)
    {
      constexpr static int SIZE_K = 1024;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 768)
    {
      constexpr static int SIZE_K = 768;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 512)
    {
      constexpr static int SIZE_K = 512;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 256)
    {
      constexpr static int SIZE_K = 256;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
  }

#if 1
  if (warp_k_id == 0 || warp_k_num == 1)
  {
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * Dtype::num2float(b_scale[min_tile_n * 4 + reg_id]);
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
          output_lds[index] = num_as_u16_bits(Dtype::float2num(value));
        }
      }
    }
  }

  __syncthreads();

  {
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if (token_index < size_m * top_k)
      {
        *reinterpret_cast<vec8_Element<Element> *>(&g_output[token_index * size_n + n_idx * 8]) =
            *reinterpret_cast<vec8_Element<Element> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
      }
    }
  }
#else

#endif
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
    uint16_t GROUP_N,
    uint16_t GROUP_K,
    int STAGES,
    bool mul_topk_weight>
__global__ void __launch_bounds__(256, 1) MOE_W8A16_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN(
    const Element *input,
    const uint32_t *__restrict__ qweight,
    Element *__restrict__ output,
    Element *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *__restrict__ sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t delta)
{
  const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
  const int bidy = blockIdx.x; // pid_n
  const int bidz = blockIdx.y; // pid_k

  if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0])
    return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144
  using Dtype = ScalarType<Element>;
  using scalar_t2 = typename ScalarType<Element>::scalar_t2;
  const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
  const int32_t delta_bidx = bidx;
  const int32_t expert_id = expert_ids[delta_bidx];                              // 专家的索引
  const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;        // 这个是对应专家的weight偏移
  const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N * 128;     // 具体偏移到对应专家的weight的某一个小的分块
  const uint32_t output_offset = bidy * BLOCK_SIZE_N;                            // 计算之后是mxn,这应该是计算输出n方向的位置
  const uint64_t weight_scale_offset = size_n * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块

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
  const int warp_k_num = BLOCK_SIZE_K / WARP_K;
  int warp_k_id = warp_id % warp_k_num;
  int warp_n_id = warp_id / warp_k_num;
  extern __shared__ uint16_t output_lds[]; // 声明lds信息
  const int *sorted_token_ids_offset = sorted_token_ids;

  union_vec_opt_w8a16_A<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
  union_vec_opt_w8a16<uint32_t, WARP_K / 16> B_int_reg[WARP_N / mfma_n][STAGES];
  Element B_scale_reg[WARP_N / mfma_n][STAGES];
  reg_bf16_fp16<scalar_t2> B_reg[WARP_N / mfma_n][STAGES][WARP_K / 32];

  auto g_qweight = qweight + qweight_offset / 4;
  auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移

  auto *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

  Element b_scale[(WARP_N / mfma_n) * 4];

  vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64 * (WARP_N / mfma_n), Element>(b_scale_ptr);
#pragma unroll
  for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
  {

    int base_off = min_tile_n * mfma_n;
    b_scale[min_tile_n * 4 + 0] = b_scale_ptr[base_off + col_id + 0];
    b_scale[min_tile_n * 4 + 1] = b_scale_ptr[base_off + col_id + 4];
    b_scale[min_tile_n * 4 + 2] = b_scale_ptr[base_off + col_id + 8];
    b_scale[min_tile_n * 4 + 3] = b_scale_ptr[base_off + col_id + 12];
  }

  floatx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
  float weight_dot_a_scale[WARP_M / mfma_m];
#pragma unroll
  for (int idx = 0; idx < WARP_M / mfma_m; idx++)
  {
    int token_index = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
    int token_index_safe = std::min(uint32_t(token_index), size_m * top_k - 1);
    weight_dot_a_scale[idx] = topk_weights[token_index_safe];
  }

  {
    if (size_k == 8192)
    {
      constexpr static int SIZE_K = 8192;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 7168)
    {
      constexpr static int SIZE_K = 7168;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 6144)
    {
      constexpr static int SIZE_K = 6144;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 5120)
    {
      constexpr static int SIZE_K = 5120;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 4096)
    {
      constexpr static int SIZE_K = 4096;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 3072)
    {
      constexpr static int SIZE_K = 3072;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 2048)
    {
      constexpr static int SIZE_K = 2048;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 1536)
    {
      constexpr static int SIZE_K = 1536;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 1024)
    {
      constexpr static int SIZE_K = 1024;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 768)
    {
      constexpr static int SIZE_K = 768;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 512)
    {
      constexpr static int SIZE_K = 512;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
    else if (size_k == 256)
    {
      constexpr static int SIZE_K = 256;
      gemm_nt_marlin_decode_w8a16<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element>(g_input, g_qweight, g_weight_scale, size_m, size_n, A_reg, B_int_reg, B_scale_reg, B_reg, C_reg, warp_id, size_k, size_k, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);
    }
  }

#if 1
  if (warp_k_id == 0 || warp_k_num == 1)
  {
    for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++)
    {
      for (int min_tile_n = 0; min_tile_n < WARP_N / mfma_n; min_tile_n++)
      {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
          float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n][reg_id] * weight_dot_a_scale[min_tile_m] * Dtype::num2float(b_scale[min_tile_n * 4 + reg_id]);
          int index = min_tile_m * mfma_m * BLOCK_SIZE_N + min_tile_n * mfma_n + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 4 + lane_id / 16 + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
          output_lds[index] = num_as_u16_bits(Dtype::float2num(value));
        }
      }
    }
  }

  __syncthreads();

  {
    const int tid = threadIdx.x;
    constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
    int m_idx = threadIdx.x / N_thread;
    int n_idx = threadIdx.x % N_thread;
    for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread)
    {
      const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

      if (token_index < size_m * top_k)
      {
        *reinterpret_cast<vec8_Element<Element> *>(&g_output[token_index * size_n + n_idx * 8]) =
            *reinterpret_cast<vec8_Element<Element> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
      }
    }
  }
#else

#endif
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a16_first_stage_decode(const GemmParams_w8a16<T> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = 16 * 1024; // max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  { // marlin版本

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.y = 1;                                                                                      // k方向

    MOE_W8A16_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                 GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.sorted_token_lens,
        params.top_k,
        params.delta);
  }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a16_second_stage_decode(const GemmParams_w8a16<T> &params)
{
  constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
  const bool mul_topk_weight = false;
  constexpr int GROUP_N = 1;
  constexpr int GROUP_K = 1;

  dim3 blockDim, gridDim;
  blockDim.x = WARP_NUM * 64;
  blockDim.y = 1;
  blockDim.z = 1;

  const int lds_size = 16 * 1024; // max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
  const int shared_mem_size = lds_size;

  const hipStream_t stream = at::cuda::getCurrentHIPStream();

  if (params.is_marlin == false)
  {
  }
  else
  { // marlin版本

    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N);                                                    // n方向
    gridDim.y = 1;                                                                                      // k方向

    MOE_W8A16_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN<T, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                   GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        params.ptr_A,
        params.ptr_B0,
        params.ptr_C,
        params.ptr_B_scale,
        params.topk_weights,
        params.sorted_token_ids,
        params.expert_ids,
        params.num_tokens_post_pad_ptr,
        params.size_m,
        params.size_n,
        params.size_k,
        params.sorted_token_lens,
        params.top_k,
        params.delta);
  }
}
