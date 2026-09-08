// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "moe_c_common.h"
#include "aiter_hip_common.h"

torch::Tensor moe_c_moe_wna16_gemm_base(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto options =
      torch::TensorOptions().dtype(input.dtype()).device(input.device());

  const int size_m = input.size(0);
  const int size_n = b_qweight.size(1);
  const int size_k = input.size(1);
  const int group_size = size_k / b_scales.size(2);
  
  // BLOCK_SIZE_K = std::min(group_size, BLOCK_SIZE_K);
  BLOCK_SIZE_K = 64;
  BLOCK_SIZE_N = 256; 
  int64_t EM = sorted_token_ids.size(0);
  if (size_m <= BLOCK_SIZE_M) {
    EM = min(EM, size_m * BLOCK_SIZE_M * top_k);
  }
  const int num_token_blocks = (EM + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;

  const uint32_t* b_qzeros_ptr;
  if (b_qzeros.has_value())
    b_qzeros_ptr = (const uint32_t*)b_qzeros.value().data_ptr<uint8_t>();
  const float* topk_weights_ptr;
  if (topk_weights.has_value())
    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();

  int groups_per_block_row = BLOCK_SIZE_K / group_size;

  TORCH_CHECK(bit == 4 || bit == 8, "bit must be 4 or 8");
  TORCH_CHECK(size_k % BLOCK_SIZE_K == 0,
              "size_k must divisible by BLOCK_SIZE_K");
  TORCH_CHECK(BLOCK_SIZE_K % group_size == 0,
              "BLOCK_SIZE_K must divisible by group_size");
  TORCH_CHECK(BLOCK_SIZE_M <= 64, "BLOCK_SIZE_M must less or equal to 64");
  TORCH_CHECK(groups_per_block_row == 1 || groups_per_block_row == 2 ||
                  groups_per_block_row == 4 || groups_per_block_row == 8,
              "BLOCK_SIZE_K // group_size must be one of [1, 2, 4, 8]");
  
  torch::Tensor output_fp32 = torch::empty(output.sizes(),output.options().dtype(torch::kFloat32));
  //   if (input.scalar_type() == at::ScalarType::Half) {
  //    half_t * d_w_out =nullptr;
  //     TOPK_SWITCH(top_k, TOPK, [&]{
  //       BLOCK_M_SWITCH(BLOCK_SIZE_M, BLOCK_SIZE_M_, [&]{
  //         BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, [&]{
  //             // BOOL_SWITCH(b_qzeros.has_value(), has_zp, [&]{
  //               BOOL_SWITCH(topk_weights.has_value(), mul_topk_weight, [&]{
  //                 GROUP_SIZE_SWITCH(group_size, GROUP_SIZE, [&]{
  //                   // run_moe_wna16_gemm<half, bit, top_k, BLOCK_SIZE_M_, BLOCK_SIZE_N, BLOCK_SIZE_K, true, mul_topk_weight, group_size>(
  //                     run_moe_wna16_gemm_base<half, 4, TOPK, BLOCK_SIZE_M_, 256, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE>(
  //                     (const half*)input.data_ptr<at::Half>(),
  //                     (float*)output_fp32.data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<uint8_t>(),
  //                     (const half*)b_scales.data_ptr<at::Half>(), 
  //                     b_qzeros_ptr,
  //                     topk_weights_ptr, 
  //                     sorted_token_ids.data_ptr<int32_t>(),
  //                     expert_ids.data_ptr<int32_t>(), 
  //                     num_tokens_post_pad.data_ptr<int32_t>(), 
  //                     num_token_blocks, 
  //                     size_m, 
  //                     size_n,
  //                     size_k
  //                   );
  //                 });
  //                 });
  //             });
  //         });
  //     });
  // } 
  // else if (input.scalar_type() == at::ScalarType::BFloat16) {
  //    __hip_bfloat16 * d_w_out =nullptr;
  //     TOPK_SWITCH(top_k, TOPK, [&]{
  //       BLOCK_M_SWITCH(BLOCK_SIZE_M, BLOCK_SIZE_M_, [&]{
  //         BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, [&]{
  //             // BOOL_SWITCH(b_qzeros.has_value(), has_zp, [&]{
  //               BOOL_SWITCH(topk_weights.has_value(), mul_topk_weight, [&]{
  //                 GROUP_SIZE_SWITCH(group_size, GROUP_SIZE, [&]{
  //                   // run_moe_wna16_gemm<half, bit, top_k, BLOCK_SIZE_M_, BLOCK_SIZE_N, BLOCK_SIZE_K, true, mul_topk_weight, group_size>(
  //                     run_moe_wna16_gemm_base<__hip_bfloat16, 4, TOPK, BLOCK_SIZE_M_, 256, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE>(
  //                     (const __hip_bfloat16*)input.data_ptr<at::BFloat16>(),
  //                     (float*)output_fp32.data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<uint8_t>(),
  //                     (const __hip_bfloat16*)b_scales.data_ptr<at::BFloat16>(), 
  //                     b_qzeros_ptr,
  //                     topk_weights_ptr, 
  //                     sorted_token_ids.data_ptr<int32_t>(),
  //                     expert_ids.data_ptr<int32_t>(), 
  //                     num_tokens_post_pad.data_ptr<int32_t>(), 
  //                     num_token_blocks, 
  //                     size_m, 
  //                     size_n,
  //                     size_k
  //                   );
  //                 });
  //                 });
  //             });
  //         });
  //     });
  // } 
  // else {
  // }
  if (input.scalar_type() == at::ScalarType::Half) {
    output.copy_(output_fp32.to(torch::kFloat16));
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    output.copy_(output_fp32.to(torch::kBFloat16));  // 转换为 BF16
  }
  return output;
}

torch::Tensor moe_c_moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_m, int64_t BLOCK_SIZE_n,
                             int64_t BLOCK_SIZE_k, int64_t kloops, int64_t nloops, int64_t bit) {
 
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  // auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  const int size_m = input.size(0);
  const int size_n = b_qweight.size(3);
  const int size_k = input.size(1);
  const int group_size = size_k / b_scales.size(2);

  int64_t BLOCK_SIZE_N = std::min(64, size_n);
  int64_t BLOCK_SIZE_K_MIN =4*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K_MAX =8*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K = std::min(BLOCK_SIZE_K_MAX,size_k);

  // int BLOCK_SIZE_M_MAX = std::min(32, size_m);
  int BLOCK_SIZE_N_MAX_roofline = 256;
  int BLOCK_SIZE_N_MAX = std::min(BLOCK_SIZE_N_MAX_roofline, size_n);

  BLOCK_SIZE_K = std::min(128, BLOCK_SIZE_K);
  int block_size_m_loops = 1;// std::min(1,BLOCK_SIZE_M_MAX/BLOCK_SIZE_m);
  int block_size_k_loops = kloops;
  int block_size_n_loops = nloops;
  // half_t * d_w_out_half =nullptr;
  // __hip_bfloat16 * d_w_out_bf =nullptr;
  // float * float_d_out = nullptr;
  // hipMalloc(&d_w_out_half, 64 * 8 * sizeof(half_t));
  // hipMalloc(&d_w_out_bf, 64 * 8 * sizeof(__hip_bfloat16));
  // hipMalloc(&float_d_out, 64 * 8 * sizeof(float));
  int64_t EM = sorted_token_ids.size(0);
  
  if (size_m <= BLOCK_SIZE_m) {
    EM = min(EM, size_m * BLOCK_SIZE_m * top_k);
  }
  const int num_token_blocks = (EM + BLOCK_SIZE_m*block_size_m_loops - 1) / (BLOCK_SIZE_m*block_size_m_loops);


  const uint32_t* b_qzeros_ptr;
  if (b_qzeros.has_value())
    b_qzeros_ptr = (const uint32_t*)b_qzeros.value().data_ptr<uint8_t>();
  const float* topk_weights_ptr;
  if (topk_weights.has_value())
    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();

  int groups_per_block_row = BLOCK_SIZE_K / group_size;
  TORCH_CHECK(bit == 4 || bit == 8, "bit must be 4 or 8");
  TORCH_CHECK(size_k % BLOCK_SIZE_K == 0,
              "size_k must divisible by BLOCK_SIZE_K");
  TORCH_CHECK(BLOCK_SIZE_K % group_size == 0,
              "BLOCK_SIZE_K must divisible by group_size");
  TORCH_CHECK(BLOCK_SIZE_m <= 64, "BLOCK_SIZE_m must less or equal to 64");
  TORCH_CHECK(groups_per_block_row == 1 || groups_per_block_row == 2 ||
                  groups_per_block_row == 4 || groups_per_block_row == 8,
              "BLOCK_SIZE_K // group_size must be one of [1, 2, 4, 8]");
  
  bool use_atomic = (size_k != BLOCK_SIZE_K*block_size_k_loops);

  std::optional<torch::Tensor> output_fp32;

  if (use_atomic){

    output_fp32 = torch::zeros(output.sizes(),output.options().dtype(torch::kFloat32));
    // output_fp32->zero_(); 
  } 

  float milliseconds = 0;
  cudaEvent_t start, stop;
  const char* find_best = std::getenv("WHICH_TO_TEST");
  if (find_best) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);        // 记录开始
  }

  // if (input.scalar_type() == at::ScalarType::Half) {
  //   BIT_SWITCH(bit, BIT, [&]{
  //     TOPK_SWITCH(top_k, TOPK, [&]{
  //       BLOCK_M_SWITCH(BLOCK_SIZE_m, BLOCK_SIZE_M_, [&]{
  //         BLOCK_N_SWITCH(BLOCK_SIZE_N, BLOCK_SIZE_N_, [&]{
  //           BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, [&]{
  //             // BOOL_SWITCH(b_qzeros.has_value(), has_zp, [&]{
  //               BOOL_SWITCH(topk_weights.has_value(), mul_topk_weight, [&]{
  //                 GROUP_SIZE_K_SWITCH(group_size, GROUP_SIZE_K, [&]{
  //                   BLOCK_SIZE_M_LOOPS_SWITCH(block_size_m_loops , BLOCK_SIZE_M_LOOPS, [&]{
  //                     BLOCK_SIZE_N_LOOPS_SWITCH(block_size_n_loops , BLOCK_SIZE_N_LOOPS, [&]{
  //                       BLOCK_SIZE_K_LOOPS_SWITCH(block_size_k_loops , BLOCK_SIZE_K_LOOPS, [&]{
  //                         BOOL_SWITCH(use_atomic , USE_ATOMIC, [&]{
  //                   run_moe_wna16_gemm<half, 4, TOPK, BLOCK_SIZE_M_, BLOCK_SIZE_N_, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE_K, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC,256>(
  //                     (const half*)input.data_ptr<at::Half>(),
  //                     // (const half*)d_input,
  //                     use_atomic ?(float*)output_fp32->data_ptr():(float*)output.data_ptr(),
  //                     // (float*)output_fp32->data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<uint32_t>(),
  //                     // (const uint32_t*)d_w_test,
  //                     // ( half_t*) d_w_out_half, /*for debug*/
  //                     // ( float*) float_d_out, /*for debug*/
  //                     (const half*)b_scales.data_ptr<at::Half>(), 
  //                     // (const half*)d_scale, 
  //                     b_qzeros_ptr,
  //                     // (const uint32_t*)d_scale,
  //                     topk_weights_ptr, 
  //                     sorted_token_ids.data_ptr<int32_t>(),
  //                     expert_ids.data_ptr<int32_t>(), 
  //                     num_tokens_post_pad.data_ptr<int32_t>(), 
  //                     // num_tokens_post_pad_value<int32_t>(),
  //                     // num_tokens_post_pad_data_ptr[0],
  //                     num_token_blocks, 
  //                     size_m, 
  //                     size_n,
  //                     size_k
  //                     ); // kernel-1 mma    
  //                   });              
  //                   });
  //                 });
  //               });
  //             });
  //           });
  //         });
  //       });
  //     });
  //   });
  // });
  // } 
  //   else if (input.scalar_type() == at::ScalarType::BFloat16) {
  //     BIT_SWITCH(bit, BIT, [&]{
  //     TOPK_SWITCH(top_k, TOPK, [&]{
  //       BLOCK_M_SWITCH(BLOCK_SIZE_m, BLOCK_SIZE_M_, [&]{
  //         BLOCK_N_SWITCH(BLOCK_SIZE_N, BLOCK_SIZE_N_, [&]{
  //           BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, [&]{
  //             // BOOL_SWITCH(b_qzeros.has_value(), has_zp, [&]{
  //               BOOL_SWITCH(topk_weights.has_value(), mul_topk_weight, [&]{
  //                 GROUP_SIZE_K_SWITCH(group_size, GROUP_SIZE_K, [&]{
  //                   BLOCK_SIZE_M_LOOPS_SWITCH(block_size_m_loops , BLOCK_SIZE_M_LOOPS, [&]{
  //                     BLOCK_SIZE_N_LOOPS_SWITCH(block_size_n_loops , BLOCK_SIZE_N_LOOPS, [&]{
  //                       BLOCK_SIZE_K_LOOPS_SWITCH(block_size_k_loops , BLOCK_SIZE_K_LOOPS, [&]{
  //                         BOOL_SWITCH(use_atomic , USE_ATOMIC, [&]{
  //                   run_moe_wna16_gemm<__hip_bfloat16, 4, TOPK, BLOCK_SIZE_M_, BLOCK_SIZE_N_, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE_K, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC,256>(
  //                     (const __hip_bfloat16*)input.data_ptr<at::BFloat16>(),
  //                     // (const half*)d_input,
  //                     use_atomic ?(float*)output_fp32->data_ptr():(float*)output.data_ptr(),
  //                     // (float*)output_fp32->data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<uint32_t>(),
  //                     // (const uint32_t*)d_w_test,
  //                     // ( __hip_bfloat16*) d_w_out_bf, /*for debug*/
  //                     // ( float*) float_d_out, /*for debug*/
  //                     (const __hip_bfloat16*)b_scales.data_ptr<at::BFloat16>(), 
  //                     // (const half*)d_scale, 
  //                     b_qzeros_ptr,
  //                     // (const uint32_t*)d_scale,
  //                     topk_weights_ptr, 
  //                     sorted_token_ids.data_ptr<int32_t>(),
  //                     expert_ids.data_ptr<int32_t>(), 
  //                     num_tokens_post_pad.data_ptr<int32_t>(), 
  //                     // num_tokens_post_pad_value<int32_t>(),
  //                     // num_tokens_post_pad_data_ptr[0],
  //                     num_token_blocks, 
  //                     size_m, 
  //                     size_n,
  //                     size_k
  //                     ); // kernel-1 mma    
  //                   });              
  //                   });
  //                 });
  //               });
  //             });
  //           });
  //         });
  //       });
  //     });
  //   });
  // });
  // } 
  // else {
  // }

  if (find_best) {
    cudaEventRecord(stop);         // 记录结束
    cudaEventSynchronize(stop);    // 等待 kernel 执行完成

    
    cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::ofstream ofs("./w4a16_kernel_1_timecost", std::ios::app); // 追加写入
    if (ofs.is_open()) {
        ofs << milliseconds << std::endl;
        ofs.close();
    }
  }

  if (use_atomic){
      if (input.scalar_type() == at::ScalarType::Half) {
        output.copy_(output_fp32->to(torch::kFloat16));
      } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        output.copy_(output_fp32->to(torch::kBFloat16));  // 转换为 BF16
      }
  }
  return output;
}

torch::Tensor moe_c_moe_wna16_gemm_2(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_m, int64_t BLOCK_SIZE_n,
                             int64_t BLOCK_SIZE_k, int64_t kloops, int64_t nloops, int64_t bit) {
 
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  // auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  const int size_m = input.size(0);
  const int size_n = b_qweight.size(1);
  const int size_k = input.size(1);
  const int group_size = size_k / b_scales.size(2);

  int64_t BLOCK_SIZE_N = std::min(64, size_n);
  int64_t BLOCK_SIZE_K_MIN =4*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K_MAX =8*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K = std::min(BLOCK_SIZE_K_MAX,size_k);

  // int BLOCK_SIZE_M_MAX = std::min(32, size_m);
  int BLOCK_SIZE_N_MAX_roofline = 256;
  int BLOCK_SIZE_N_MAX = std::min(BLOCK_SIZE_N_MAX_roofline, size_n);

  BLOCK_SIZE_K = std::min(128, BLOCK_SIZE_K);

  int block_size_m_loops = 1;// std::min(1,BLOCK_SIZE_M_MAX/BLOCK_SIZE_m);
  int block_size_k_loops = kloops;
  int block_size_n_loops = nloops;
  int64_t EM = sorted_token_ids.size(0);
  
  if (size_m <= BLOCK_SIZE_m) {
    EM = min(EM, size_m * BLOCK_SIZE_m * top_k);
  }
  const int num_token_blocks = (EM + BLOCK_SIZE_m*block_size_m_loops - 1) / (BLOCK_SIZE_m*block_size_m_loops);

  const uint32_t* b_qzeros_ptr;
  if (b_qzeros.has_value())
    b_qzeros_ptr = (const uint32_t*)b_qzeros.value().data_ptr<uint8_t>();
  const float* topk_weights_ptr;
  if (topk_weights.has_value())
    topk_weights_ptr = (const float*)topk_weights.value().data_ptr();

  int groups_per_block_row = BLOCK_SIZE_K / group_size;
  TORCH_CHECK(bit == 4 || bit == 8, "bit must be 4 or 8");
  TORCH_CHECK(size_k % BLOCK_SIZE_K == 0,
              "size_k must divisible by BLOCK_SIZE_K");
  TORCH_CHECK(BLOCK_SIZE_K % group_size == 0,
              "BLOCK_SIZE_K must divisible by group_size");
  TORCH_CHECK(BLOCK_SIZE_m <= 64, "BLOCK_SIZE_m must less or equal to 64");
  TORCH_CHECK(groups_per_block_row == 1 || groups_per_block_row == 2 ||
                  groups_per_block_row == 4 || groups_per_block_row == 8,
              "BLOCK_SIZE_K // group_size must be one of [1, 2, 4, 8]");
  
  bool use_atomic = (size_k != BLOCK_SIZE_K*block_size_k_loops);

  std::optional<torch::Tensor> output_fp32;

  if (use_atomic){

    output_fp32 = torch::zeros(output.sizes(),output.options().dtype(torch::kFloat32));
    // output_fp32->zero_(); 
  } 

  float milliseconds = 0;
  cudaEvent_t start, stop;
  const char* find_best = std::getenv("WHICH_TO_TEST");
  if (find_best) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);        // 记录开始
  }

  // if (input.scalar_type() == at::ScalarType::Half) {
  //   BIT_SWITCH(bit, BIT, [&]{
  //     TOPK_SWITCH(top_k, TOPK, [&]{
  //       BLOCK_M_SWITCH(BLOCK_SIZE_m, BLOCK_SIZE_M_, [&]{
  //         BLOCK_N_SWITCH(BLOCK_SIZE_N, BLOCK_SIZE_N_, [&]{
  //           BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, [&]{
  //             // BOOL_SWITCH(b_qzeros.has_value(), has_zp, [&]{
  //               BOOL_SWITCH(topk_weights.has_value(), mul_topk_weight, [&]{
  //                 GROUP_SIZE_K_SWITCH(group_size, GROUP_SIZE_K, [&]{
  //                   BLOCK_SIZE_M_LOOPS_SWITCH(block_size_m_loops , BLOCK_SIZE_M_LOOPS, [&]{
  //                     BLOCK_SIZE_N_LOOPS_SWITCH(block_size_n_loops , BLOCK_SIZE_N_LOOPS, [&]{
  //                       BLOCK_SIZE_K_LOOPS_SWITCH(block_size_k_loops , BLOCK_SIZE_K_LOOPS, [&]{
  //                         BOOL_SWITCH(use_atomic , USE_ATOMIC, [&]{
  //                   run_moe_wna16_gemm_2<half, 4, TOPK, BLOCK_SIZE_M_, BLOCK_SIZE_N_, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE_K, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC,256>(
  //                     (const half*)input.data_ptr<at::Half>(),
  //                     // (const half*)d_input,
  //                     use_atomic ?(float*)output_fp32->data_ptr():(float*)output.data_ptr(),
  //                     // (float*)output_fp32->data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<uint8_t>(),
  //                     // (const uint32_t*)d_w_test,
  //                     // ( half_t*) d_w_out_half, /*for debug*/
  //                     // ( float*) float_d_out, /*for debug*/
  //                     (const half*)b_scales.data_ptr<at::Half>(), 
  //                     // (const half*)d_scale, 
  //                     b_qzeros_ptr,
  //                     // (const uint32_t*)d_scale,
  //                     topk_weights_ptr, 
  //                     sorted_token_ids.data_ptr<int32_t>(),
  //                     expert_ids.data_ptr<int32_t>(), 
  //                     num_tokens_post_pad.data_ptr<int32_t>(), 
  //                     // num_tokens_post_pad_value<int32_t>(),
  //                     // num_tokens_post_pad_data_ptr[0],
  //                     num_token_blocks, 
  //                     size_m, 
  //                     size_n,
  //                     size_k
  //                     ); // kernel-1 mma    
  //                   });              
  //                   });
  //                 });
  //               });
  //             });
  //           });
  //         });
  //       });
  //     });
  //   });
  // });
  // } 
  //   else if (input.scalar_type() == at::ScalarType::BFloat16) {
  //     BIT_SWITCH(bit, BIT, [&]{
  //     TOPK_SWITCH(top_k, TOPK, [&]{
  //       BLOCK_M_SWITCH(BLOCK_SIZE_m, BLOCK_SIZE_M_, [&]{
  //         BLOCK_N_SWITCH(BLOCK_SIZE_N, BLOCK_SIZE_N_, [&]{
  //           BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, [&]{
  //             // BOOL_SWITCH(b_qzeros.has_value(), has_zp, [&]{
  //               BOOL_SWITCH(topk_weights.has_value(), mul_topk_weight, [&]{
  //                 GROUP_SIZE_K_SWITCH(group_size, GROUP_SIZE_K, [&]{
  //                   BLOCK_SIZE_M_LOOPS_SWITCH(block_size_m_loops , BLOCK_SIZE_M_LOOPS, [&]{
  //                     BLOCK_SIZE_N_LOOPS_SWITCH(block_size_n_loops , BLOCK_SIZE_N_LOOPS, [&]{
  //                       BLOCK_SIZE_K_LOOPS_SWITCH(block_size_k_loops , BLOCK_SIZE_K_LOOPS, [&]{
  //                         BOOL_SWITCH(use_atomic , USE_ATOMIC, [&]{
  //                   run_moe_wna16_gemm_2<__hip_bfloat16, 4, TOPK, BLOCK_SIZE_M_, BLOCK_SIZE_N_, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE_K, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC,256>(
  //                     (const __hip_bfloat16*)input.data_ptr<at::BFloat16>(),
  //                     // (const half*)d_input,
  //                     use_atomic ?(float*)output_fp32->data_ptr():(float*)output.data_ptr(),
  //                     // (float*)output_fp32->data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<uint8_t>(),
  //                     // (const uint32_t*)d_w_test,
  //                     // ( __hip_bfloat16*) d_w_out_bf, /*for debug*/
  //                     // ( float*) float_d_out, /*for debug*/
  //                     (const __hip_bfloat16*)b_scales.data_ptr<at::BFloat16>(), 
  //                     // (const half*)d_scale, 
  //                     b_qzeros_ptr,
  //                     // (const uint32_t*)d_scale,
  //                     topk_weights_ptr, 
  //                     sorted_token_ids.data_ptr<int32_t>(),
  //                     expert_ids.data_ptr<int32_t>(), 
  //                     num_tokens_post_pad.data_ptr<int32_t>(), 
  //                     // num_tokens_post_pad_value<int32_t>(),
  //                     // num_tokens_post_pad_data_ptr[0],
  //                     num_token_blocks, 
  //                     size_m, 
  //                     size_n,
  //                     size_k
  //                     ); // kernel-1 mma    
  //                   });              
  //                   });
  //                 });
  //               });
  //             });
  //           });
  //         });
  //       });
  //     });
  //   });
  // });
  // } 
  // else {
  // }

  if (find_best) {
    cudaEventRecord(stop);         // 记录结束
    cudaEventSynchronize(stop);    // 等待 kernel 执行完成

    
    cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::ofstream ofs("./w4a16_kernel_2_timecost", std::ios::app); // 追加写入
    if (ofs.is_open()) {
        ofs << milliseconds << std::endl;
        ofs.close();
    }
  }

  if (use_atomic){
      if (input.scalar_type() == at::ScalarType::Half) {
        output.copy_(output_fp32->to(torch::kFloat16));
      } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        output.copy_(output_fp32->to(torch::kBFloat16));  // 转换为 BF16
      }
  }
  return output;
}

namespace {

template <typename T>
auto& w16a16_gemm1_prefill_map() {
  if constexpr (std::is_same_v<T, bhalf_t>) {
    return at::native::kernel_maps_gemm1_prefill_marlin_w16a16_bhalf_t;
  } else {
    return at::native::kernel_maps_gemm1_prefill_marlin_w16a16_half;
  }
}

template <typename T>
auto& w16a16_gemm1_decode_map() {
  if constexpr (std::is_same_v<T, bhalf_t>) {
    return at::native::kernel_maps_gemm1_decode_marlin_w16a16_bhalf_t;
  } else {
    return at::native::kernel_maps_gemm1_decode_marlin_w16a16_half;
  }
}

template <typename T>
auto& w16a16_gemm2_prefill_map() {
  if constexpr (std::is_same_v<T, bhalf_t>) {
    return at::native::kernel_maps_gemm2_prefill_marlin_w16a16_bhalf_t;
  } else {
    return at::native::kernel_maps_gemm2_prefill_marlin_w16a16_half;
  }
}

template <typename T>
auto& w16a16_gemm2_decode_map() {
  if constexpr (std::is_same_v<T, bhalf_t>) {
    return at::native::kernel_maps_gemm2_decode_marlin_w16a16_bhalf_t;
  } else {
    return at::native::kernel_maps_gemm2_decode_marlin_w16a16_half;
  }
}

template <typename T>
void dispatch_w16a16_marlin_gemm(
    bool first_stage,
    const torch::Tensor& input,
    const torch::Tensor& b_qweight,
    torch::Tensor& output_alias,
    const float* topk_weights_ptr,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& expert_ids,
    const torch::Tensor& num_tokens_post_pad,
    int size_m,
    int size_n,
    int size_k,
    int size_kb,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int experts) {
  const int64_t sorted_token_lens = sorted_token_ids.size(0);
  at::native::GemmParams3<T> params(
      reinterpret_cast<const T*>(input.data_ptr()),
      reinterpret_cast<const T*>(b_qweight.data_ptr()),
      reinterpret_cast<T*>(output_alias.data_ptr()),
      topk_weights_ptr,
      sorted_token_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      0,
      num_tokens_post_pad.data_ptr<int32_t>(),
      static_cast<uint32_t>(size_m),
      static_cast<uint32_t>(size_n),
      static_cast<uint32_t>(size_k),
      static_cast<uint32_t>(size_kb),
      static_cast<uint32_t>(sorted_token_lens),
      static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(delta),
      static_cast<uint32_t>(experts));

  const int mode_i = static_cast<int>(mode);
  if (first_stage) {
    if (mode_i < 300) {
      auto& kernel_map = w16a16_gemm1_prefill_map<T>();
      auto it = kernel_map.find(mode_i);
      if (it != kernel_map.end()) {
        it->second(params);
      } else {
        TORCH_CHECK(false,
                    "unsupported w16a16 GEMM1 prefill kernel mode: ",
                    mode_i);
      }
    } else {
      auto& kernel_map = w16a16_gemm1_decode_map<T>();
      auto it = kernel_map.find(mode_i);
      if (it != kernel_map.end()) {
        it->second(params);
      } else {
        TORCH_CHECK(false,
                    "unsupported w16a16 GEMM1 decode kernel mode: ",
                    mode_i);
      }
    }
  } else {
    if (mode_i < 300) {
      auto& kernel_map = w16a16_gemm2_prefill_map<T>();
      auto it = kernel_map.find(mode_i);
      if (it != kernel_map.end()) {
        it->second(params);
      } else {
        TORCH_CHECK(false,
                    "unsupported w16a16 GEMM2 prefill kernel mode: ",
                    mode_i);
      }
    } else {
      auto& kernel_map = w16a16_gemm2_decode_map<T>();
      auto it = kernel_map.find(mode_i);
      if (it != kernel_map.end()) {
        it->second(params);
      } else {
        TORCH_CHECK(false,
                    "unsupported w16a16 GEMM2 decode kernel mode: ",
                    mode_i);
      }
    }
  }
}

static inline uint32_t w16a16_asm_divide(uint32_t x, uint32_t size) {
  return (x + size - 1) / size;
}

template <typename T>
struct W16A16MarlinAsmArgs {
  uint32_t numWorkGroups0;
  uint32_t numWorkGroups1;
  T* ptr_C;
  const bhalf_t* ptr_A;
  const bhalf_t* ptr_B;
  float* ptr_A_scale;
  float* ptr_B_scale;
  const float* topk_weights;
  const int32_t* sorted_token_ids;
  const int32_t* expert_ids;
  const int32_t* num_tokens_post_pad_ptr;
  uint32_t experts_num;
  uint32_t size_m;
  uint32_t size_n;
  uint32_t size_k;
  uint32_t stride_asm;
  uint32_t stride_ask;
  uint32_t stride_bse;
  uint32_t stride_bsn;
  uint32_t stride_bsk;
  uint32_t sorted_token_lens;
  uint32_t topk;
  float topk_rcip;
  float delta_rcip;
  void* debugBuffer;
};

template <int BLOCKM, int BLOCKN, int BLOCKK, typename OutputType>
void launch_w16a16_marlin_asm(
    const torch::Tensor& input,
    const torch::Tensor& b_qweight,
    const torch::Tensor& output,
    const std::optional<torch::Tensor>& topk_weights,
    const torch::Tensor& sorted_token_ids,
    const torch::Tensor& expert_ids,
    const torch::Tensor& num_tokens_post_pad,
    uint32_t top_k,
    uint32_t delta,
    uint32_t experts_num) {
  const uint32_t size_m = static_cast<uint32_t>(input.size(0));
  const uint32_t size_k = static_cast<uint32_t>(input.size(1));
  const uint32_t size_n = static_cast<uint32_t>(
      b_qweight.size(2) * b_qweight.size(1) / input.size(1));
  const bool first_stage = !topk_weights.has_value();
  const float* topk_weights_ptr = first_stage
      ? nullptr
      : static_cast<const float*>(topk_weights.value().data_ptr());
  const uint32_t sorted_token_lens =
      static_cast<uint32_t>(sorted_token_ids.size(0));

  size_t localWorkSize[3] = {768, 1, 1};
  size_t globalWorkSize[3] = {
      w16a16_asm_divide(size_n, BLOCKN),
      1,
      w16a16_asm_divide(sorted_token_lens, BLOCKM),
  };

  W16A16MarlinAsmArgs<OutputType> args;
  args.numWorkGroups0 = static_cast<uint32_t>(globalWorkSize[0]);
  args.numWorkGroups1 = static_cast<uint32_t>(globalWorkSize[2]);
  args.ptr_C = static_cast<OutputType*>(output.data_ptr());
  args.ptr_A = reinterpret_cast<const bhalf_t*>(b_qweight.data_ptr());
  args.ptr_B = reinterpret_cast<const bhalf_t*>(input.data_ptr());
  args.ptr_A_scale = nullptr;
  args.ptr_B_scale = nullptr;
  args.topk_weights = topk_weights_ptr;
  args.sorted_token_ids = sorted_token_ids.data_ptr<int32_t>();
  args.expert_ids = expert_ids.data_ptr<int32_t>();
  args.num_tokens_post_pad_ptr = num_tokens_post_pad.data_ptr<int32_t>();
  args.experts_num = experts_num;
  args.size_m = size_m;
  args.size_n = size_n;
  args.size_k = size_k;
  args.stride_asm = 0;
  args.stride_ask = 0;
  args.stride_bse = 0;
  args.stride_bsn = 0;
  args.stride_bsk = 0;
  args.sorted_token_lens = sorted_token_lens;
  args.topk = top_k;
  args.topk_rcip = 1.0f / static_cast<float>(top_k);
  args.delta_rcip = 1.0f / static_cast<float>(delta);
  args.debugBuffer = nullptr;

  char funcName[1024];
  char coFile[1024];
  std::memset(funcName, 0, sizeof(funcName));
  std::memset(coFile, 0, sizeof(coFile));
  if (output.scalar_type() == at::ScalarType::Half) {
    std::snprintf(
        funcName,
        sizeof(funcName),
        first_stage
            ? "MOE_W16A16_FP16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM1_UP"
            : "MOE_W16A16_FP16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM1_DOWN",
        BLOCKM,
        BLOCKN,
        BLOCKK);
    std::snprintf(
        coFile,
        sizeof(coFile),
        first_stage
            ? "w16a16_new/moe_w16a16_marlin_%dx%dx%d_TN_FP16_UP.co"
            : "w16a16_new/moe_w16a16_marlin_%dx%dx%d_TN_FP16_DOWN.co",
        BLOCKM,
        BLOCKN,
        BLOCKK);
  } else if (output.scalar_type() == at::ScalarType::BFloat16) {
    std::snprintf(
        funcName,
        sizeof(funcName),
        first_stage
            ? "MOE_W16A16_BF16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM1_UP"
            : "MOE_W16A16_BF16_PERCHANNEL_MARLIN_ASM_TN_MT%dx%dx%d_WGM1_DOWN",
        BLOCKM,
        BLOCKN,
        BLOCKK);
    std::snprintf(
        coFile,
        sizeof(coFile),
        first_stage
            ? "w16a16_new/moe_w16a16_marlin_%dx%dx%d_TN_BF16_UP.co"
            : "w16a16_new/moe_w16a16_marlin_%dx%dx%d_TN_BF16_DOWN.co",
        BLOCKM,
        BLOCKN,
        BLOCKK);
  } else {
    TORCH_CHECK(false, "moe_marlin_w16a16 only supports Float16/BFloat16 output");
  }

  size_t argsSize = sizeof(args);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
  if (first_stage) {
    static AiterAsmKernel firstStage(funcName, coFile);
    firstStage.launch_kernel({
        &args,
        &argsSize,
        static_cast<int>(globalWorkSize[0]),
        static_cast<int>(globalWorkSize[1]),
        static_cast<int>(globalWorkSize[2]),
        static_cast<int>(localWorkSize[0]),
        static_cast<int>(localWorkSize[1]),
        static_cast<int>(localWorkSize[2]),
        stream});
  } else {
    static AiterAsmKernel secondStage(funcName, coFile);
    secondStage.launch_kernel({
        &args,
        &argsSize,
        static_cast<int>(globalWorkSize[0]),
        static_cast<int>(globalWorkSize[1]),
        static_cast<int>(globalWorkSize[2]),
        static_cast<int>(localWorkSize[0]),
        static_cast<int>(localWorkSize[1]),
        static_cast<int>(localWorkSize[2]),
        stream});
  }
}

} // namespace

torch::Tensor moe_c_moe_gemm_marlin_w16a16(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  TORCH_CHECK(mode < 400 || mode == 421, "moe_c_moe_gemm_marlin_w16a16 only supports Marlin modes");

  const int size_m = input.size(0);
  const int experts = b_qweight.size(0);
  const int size_k = input.size(1);
  const bool logical_weight_shape = b_qweight.size(2) == size_k;
  const int size_n =
      logical_weight_shape ? b_qweight.size(1) * 16 : b_qweight.size(2);
  const int size_kb =
      logical_weight_shape ? b_qweight.size(2) / 16 : b_qweight.size(1);
  const bool first_stage = !topk_weights.has_value();
  const float* topk_weights_ptr = first_stage
      ? nullptr
      : static_cast<const float*>(topk_weights.value().data_ptr());
  torch::Tensor output_alias = output.alias();

  if (input.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_w16a16_marlin_gemm<bhalf_t>(
        first_stage,
        input,
        b_qweight,
        output_alias,
        topk_weights_ptr,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        size_m,
        size_n,
        size_k,
        size_kb,
        top_k,
        mode,
        delta,
        experts);
  } else if (input.scalar_type() == at::ScalarType::Half) {
    dispatch_w16a16_marlin_gemm<half>(
        first_stage,
        input,
        b_qweight,
        output_alias,
        topk_weights_ptr,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        size_m,
        size_n,
        size_k,
        size_kb,
        top_k,
        mode,
        delta,
        experts);
  } else {
    TORCH_CHECK(false, "moe_c_moe_gemm_marlin_w16a16 only supports BFloat16/Float16");
  }

  return output;
}

torch::Tensor moe_c_moe_gemm_marlin_w16a16_asm(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  TORCH_CHECK(mode == 1000, "moe_c_moe_gemm_marlin_w16a16_asm only supports mode=1000");
  const uint32_t experts_num = static_cast<uint32_t>(b_qweight.size(0));

  if (output.scalar_type() == at::ScalarType::Half) {
    launch_w16a16_marlin_asm<128, 256, 64, half>(
        input,
        b_qweight,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        static_cast<uint32_t>(top_k),
        static_cast<uint32_t>(delta),
        experts_num);
  } else if (output.scalar_type() == at::ScalarType::BFloat16) {
    launch_w16a16_marlin_asm<128, 256, 64, bhalf_t>(
        input,
        b_qweight,
        output,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        static_cast<uint32_t>(top_k),
        static_cast<uint32_t>(delta),
        experts_num);
  } else {
    TORCH_CHECK(false, "moe_c_moe_gemm_marlin_w16a16_asm only supports BFloat16/Float16");
  }

  return output;
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def(
      "moe_c_moe_wna16_gemm("
      "Tensor input, Tensor! output, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, "
      "int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
      "int kloops, int nloops, int bit) -> Tensor");
  m.impl("moe_c_moe_wna16_gemm", torch::kCUDA, &moe_c_moe_wna16_gemm);

  m.def(
      "moe_c_moe_wna16_gemm_2("
      "Tensor input, Tensor! output, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, "
      "int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
      "int kloops, int nloops, int bit) -> Tensor");
  m.impl("moe_c_moe_wna16_gemm_2", torch::kCUDA, &moe_c_moe_wna16_gemm_2);

  m.def(
      "moe_c_moe_wna16_gemm_base("
      "Tensor input, Tensor! output, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, "
      "int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit) -> Tensor");
  m.impl(
      "moe_c_moe_wna16_gemm_base",
      torch::kCUDA,
      &moe_c_moe_wna16_gemm_base);

  m.def(
      "moe_c_moe_gemm_marlin_w16a16("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor? topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, "
      "int top_k, int mode, int delta) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_w16a16",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_w16a16);

  m.def(
      "moe_c_moe_gemm_marlin_w16a16_asm("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor? topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, "
      "int top_k, int mode, int delta) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_w16a16_asm",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_w16a16_asm);
}
