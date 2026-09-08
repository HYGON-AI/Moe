#include "moe_c_common.h"

torch::Tensor moe_c_moe_gemm_marlin_w8a16(torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor b_scale,
  std::optional<torch::Tensor> topk_weights,
  torch::Tensor sorted_token_ids, 
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_pad, 
  int64_t top_k, // gemm1为topk  gemm2为1  因为gemm1输入为[m, k]  gemm2输入为[m*topk, k]
  int64_t mode,
  int64_t delta
  ) {
    const int size_m = input.size(0); 
    const int EXPERTS = b_qweight.size(0);
    const int size_k = input.size(1);
    const int size_n = b_scale.size(1); //modifyy
    const int topk_size = output.size(1); // 输出为[m, topk_size, n]
    // const int stride_asm = a_scale.stride(0);
    // const int stride_ask = a_scale.stride(1);
    // const int stride_bse = b_scale.size(1);
    // const int stride_bsn = b_scale.size(2); 
    // const int stride_bsk = b_scale.size(2);
    // const uint32_t* b_zeros_ptr;
    // if (b_zeros.has_value())
    // b_zeros_ptr = (const uint32_t*)b_zeros.value().data_ptr<uint8_t>();
    // 单行printf打印所有步长变量，带标签便于识别
    //    stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk);
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    bool is_marlin = true; // weight为[E, N, K]时 代表不进行重排
    bool first_stage = true;
    torch::Tensor output_alias = output.alias();

    const float* topk_weights_ptr = nullptr; // 第一阶段这里为null
    if (topk_weights.has_value()){
      topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
      first_stage = false;
    }

    int num_pad = 0;
    if (input.scalar_type() == at::ScalarType::Half){
      if(first_stage){
     
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        GemmParams_w8a16<half> params_in(
          (half*)input.data_ptr<at::Half>(), 
          (uint32_t*)b_qweight.data_ptr<uint32_t>(), 
          (half*)output_alias.data_ptr<at::Half>(),
          (half*)b_scale.data_ptr<at::Half>(),  
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, //num_tokens_post_pad[0].item<int>(), //这里获取值 会造成device->host的拷贝和一部分空泡
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          // stride_asm,
          // stride_ask,
          // stride_bse,
          // stride_bsn,
          // stride_bsk,
          EM,
          top_k,
          delta,
          is_marlin
        );

        if(mode >= 500){
          // auto it = kernel_maps_gemm1_prefill.find(mode);
          // if (it != kernel_maps_gemm1_prefill.end()) {
          //     it->second(params_in);
          // } else {
          // }
        }else{ //decode 
          auto it = kernel_maps_gemm1_decode_w8a16<half>.find(mode);
          if (it != kernel_maps_gemm1_decode_w8a16<half>.end() ) {
            float milliseconds = 0;
            cudaEvent_t start, stop;
            const char* find_best = std::getenv("WHICH_TO_TEST");
            if (find_best) {
              cudaEventCreate(&start);
              cudaEventCreate(&stop);
              cudaEventRecord(start);        // 记录开始
            }

            it->second(params_in);

            if (find_best) {
              cudaEventRecord(stop);         // 记录结束
              cudaEventSynchronize(stop);    // 等待 kernel 执行完成

              
              cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间
              
              cudaEventDestroy(start);
              cudaEventDestroy(stop);

              std::ofstream ofs("./w8a16_kernel_1_timecost", std::ios::app); // 追加写入
              if (ofs.is_open()) {
                  ofs << milliseconds << std::endl;
                  ofs.close();
              }
            }
              
          } else {
          }
        }
        
        
      }else{ //gemm2
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        GemmParams_w8a16<half> params_in(
          (half*)input.data_ptr<at::Half>(), 
          (uint32_t*)b_qweight.data_ptr<uint32_t>(), 
          (half*)output_alias.data_ptr<at::Half>(),
          (half*)b_scale.data_ptr<at::Half>(),  
          topk_weights_ptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(), 
          num_pad, //num_tokens_post_pad[0].item<int>(), //这里获取值 会造成device->host的拷贝和一部分空泡
          num_tokens_post_pad.data_ptr<int32_t>(),
          size_m,
          size_n,
          size_k,
          // stride_asm,
          // stride_ask,
          // stride_bse,
          // stride_bsn,
          // stride_bsk,
          EM,
          top_k,
          delta,
          is_marlin
        );

        if(mode >= 500){
        //   auto it = kernel_maps_gemm2_prefill.find(mode);
        //   if (it != kernel_maps_gemm2_prefill.end()) {
        //       it->second(params_in);
        //   } else {
        //   }
        }else
        {
          
          auto it = kernel_maps_gemm2_decode_w8a16<half>.find(mode);
          if (it != kernel_maps_gemm2_decode_w8a16<half>.end() ) {
            float milliseconds = 0;
            cudaEvent_t start, stop;
            const char* find_best = std::getenv("WHICH_TO_TEST");
            if (find_best) {
              cudaEventCreate(&start);
              cudaEventCreate(&stop);
              cudaEventRecord(start);        // 记录开始
            }

            it->second(params_in);

            if (find_best) {
            cudaEventRecord(stop);         // 记录结束
            cudaEventSynchronize(stop);    // 等待 kernel 执行完成

            
            cudaEventElapsedTime(&milliseconds, start, stop); // 计算时间
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            std::ofstream ofs("./w8a16_kernel_2_timecost", std::ios::app); // 追加写入
            if (ofs.is_open()) {
                ofs << milliseconds << std::endl;
                ofs.close();
              }
            }
          } else {

          }

        }
              // hipDeviceSynchronize();

      }
    } else if (input.scalar_type() == at::ScalarType::BFloat16){
      int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
      GemmParams_w8a16<__hip_bfloat16> params_in(
        (__hip_bfloat16*)input.data_ptr<at::BFloat16>(),
        (uint32_t*)b_qweight.data_ptr<uint32_t>(),
        (__hip_bfloat16*)output_alias.data_ptr<at::BFloat16>(),
        (__hip_bfloat16*)b_scale.data_ptr<at::BFloat16>(),
        topk_weights_ptr,
        sorted_token_ids.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_pad, //num_tokens_post_pad[0].item<int>(), //这里获取值 会造成device->host的拷贝和一部分空泡
        num_tokens_post_pad.data_ptr<int32_t>(),
        size_m,
        size_n,
        size_k,
        EM,
        top_k,
        delta,
        is_marlin
      );

      if(mode < 500){
        if(first_stage){
          auto it = kernel_maps_gemm1_decode_w8a16<__hip_bfloat16>.find(mode);
          if (it != kernel_maps_gemm1_decode_w8a16<__hip_bfloat16>.end() ) {
            it->second(params_in);
          }
        } else {
          auto it = kernel_maps_gemm2_decode_w8a16<__hip_bfloat16>.find(mode);
          if (it != kernel_maps_gemm2_decode_w8a16<__hip_bfloat16>.end() ) {
            it->second(params_in);
          }
        }
      }
    } else {
      TORCH_CHECK(false, "moe_w8a16_gemm only supports float16 and bfloat16 input");
    }
    return output;
}

torch::Tensor moe_c_moe_w8a16_gemm_awq(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_m, int64_t BLOCK_SIZE_n,
                             int64_t BLOCK_SIZE_k, int64_t bit) {
 
  // const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  // auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());

  const int size_m = input.size(0);
  const int size_n = b_qweight.size(1);
  const int size_k = input.size(1);
  const int group_size = size_k / b_scales.size(2);
  /*经验值4-8个block，lds为64k，左矩阵BM*BK*2 范围为8k-16k， 所以BM*BN应在4k-8k*/
  int64_t BLOCK_SIZE_N = std::min(64, size_n);
  int64_t BLOCK_SIZE_K_MIN =4*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K_MAX =8*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K = std::min(BLOCK_SIZE_K_MAX,size_k);

  int BLOCK_SIZE_M_MAX = std::min(16, size_m);
  int BLOCK_SIZE_N_MAX_roofline = 256;
  int BLOCK_SIZE_N_MAX = std::min(BLOCK_SIZE_N_MAX_roofline, size_n);

  BLOCK_SIZE_K = std::min(128, BLOCK_SIZE_K);

  int block_size_m_loops = 1;// std::min(1,BLOCK_SIZE_M_MAX/BLOCK_SIZE_m);
  int block_size_k_loops = std::min(size_k/BLOCK_SIZE_K, 2);
  int block_size_n_loops = BLOCK_SIZE_N_MAX/BLOCK_SIZE_N;

  half_t * d_w_out =nullptr;
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
  //  hipDeviceSynchronize();
  // hipEventRecord(stop, 0 );
  // hipEventSynchronize( stop );

  // float ave_time;
  // hipEventElapsedTime( &ave_time,start, stop );

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
  //                   run_moe_wna16_gemm_awq<half, 8, TOPK, BLOCK_SIZE_M_, BLOCK_SIZE_N_, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE_K, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC,256>(
  //                     (const half*)input.data_ptr<at::Half>(),
  //                     // (const half*)d_input,
  //                     use_atomic ?(float*)output_fp32->data_ptr():(float*)output.data_ptr(),
  //                     // (float*)output_fp32->data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<uint8_t>(),
  //                     // (const uint32_t*)d_w_test,
  //                     ( half_t*) d_w_out, /*for debug*/
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
  //                     // run_moe_wna16_gemm_blockwise_<half, BIT, TOPK, BLOCK_SIZE_M_, BLOCK_SIZE_N_, BLOCK_SIZE_K_, true, mul_topk_weight, GROUP_SIZE, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC,256>(
  //                     // (const half*)input.data_ptr<at::Half>(),
  //                     // // (const half*)d_input,
  //                     // use_atomic ?(float*)output_fp32->data_ptr():(float*)output.data_ptr(),
  //                     // // (float*)output_fp32->data_ptr(),
  //                     // (const uint32_t*)b_qweight.data_ptr<uint8_t>(),
  //                     // // (const uint32_t*)d_w_test,
  //                     // ( half_t*) d_w_out, /*for debug*/
  //                     // (const half*)b_scales.data_ptr<at::Half>(), 
  //                     // // (const half*)d_scale, 
  //                     // b_qzeros_ptr,
  //                     // // (const uint32_t*)d_scale,
  //                     // topk_weights_ptr, 
  //                     // sorted_token_ids.data_ptr<int32_t>(),
  //                     // expert_ids.data_ptr<int32_t>(), 
  //                     // num_tokens_post_pad.data_ptr<int32_t>(), 
  //                     // // num_tokens_post_pad_value<int32_t>(),
  //                     // // num_tokens_post_pad_data_ptr[0],
  //                     // num_token_blocks, 
  //                     // size_m, 
  //                     // size_n,
  //                     // size_k
  //                     // ); // kernel-1 mma    
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

  // hipEvent_t  start, stop;
  // hipEventCreate(&start);
  // hipEventCreate(&stop);
  // hipEventRecord(start, 0);
  
  if (use_atomic){
    output.copy_(output_fp32->to(torch::kFloat16));
  }

  // hipDeviceSynchronize();
  // hipEventRecord(stop, 0 );
  // hipEventSynchronize( stop );

  // float ave_time;
  // hipEventElapsedTime( &ave_time,start, stop );

  return output;
  
}


torch::Tensor moe_c_moe_w8a16_gemm_block_wise(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t group_size_n, int64_t group_size_k, int64_t top_k,
                             int64_t BLOCK_SIZE_m, int64_t BLOCK_SIZE_n,
                             int64_t BLOCK_SIZE_k, int64_t bit) {
  // const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  // auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  
  

  const int size_m = input.size(0);
  const int size_n = b_qweight.size(1);
  const int size_k = input.size(1);
  // const int group_size = size_k / b_scales.size(2);

  /*经验值4-8个block，lds为64k，左矩阵BM*BK*2 范围为8k-16k， 所以BM*BN应在4k-8k*/

  int64_t BLOCK_SIZE_N = std::min(64, size_n);
  int64_t BLOCK_SIZE_K_MIN =4*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K_MAX =8*1024/BLOCK_SIZE_m;
  int64_t BLOCK_SIZE_K = std::min(BLOCK_SIZE_K_MAX,size_k);

  int BLOCK_SIZE_M_MAX = std::min(16, size_m);
  int BLOCK_SIZE_N_MAX_roofline = 256;
  int BLOCK_SIZE_N_MAX = std::min(BLOCK_SIZE_N_MAX_roofline, size_n);

  BLOCK_SIZE_K = std::min(128, BLOCK_SIZE_K);

  int block_size_m_loops = 1;// std::min(1,BLOCK_SIZE_M_MAX/BLOCK_SIZE_m);
  int block_size_k_loops = std::min(size_k/BLOCK_SIZE_K, 2);
  int block_size_n_loops = BLOCK_SIZE_N_MAX/BLOCK_SIZE_N;

  half_t * d_w_out =nullptr;
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

  int groups_per_block_row = BLOCK_SIZE_K / group_size_k;
  TORCH_CHECK(bit == 4 || bit == 8, "bit must be 4 or 8");
  TORCH_CHECK(size_k % BLOCK_SIZE_K == 0,
              "size_k must divisible by BLOCK_SIZE_K");
  TORCH_CHECK(BLOCK_SIZE_K % group_size_k == 0,
              "BLOCK_SIZE_K must divisible by group_size_k");
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

  // if (input.scalar_type() == at::ScalarType::Half) {

  //   BIT_SWITCH(bit, BIT, [&]{
  //     TOPK_SWITCH(top_k, TOPK, [&]{
  //       BLOCK_M_SWITCH(BLOCK_SIZE_m, BLOCK_SIZE_M_, [&]{
  //         BLOCK_N_SWITCH(BLOCK_SIZE_N, BLOCK_SIZE_N_, [&]{
  //           BLOCK_K_SWITCH(BLOCK_SIZE_K, BLOCK_SIZE_K_, [&]{
  //             // BOOL_SWITCH(b_qzeros.has_value(), has_zp, [&]{
  //               BOOL_SWITCH(topk_weights.has_value(), mul_topk_weight, [&]{
  //                 GROUP_SIZE_N_SWITCH(group_size_n, GROUP_SIZE_N, [&]{
  //                   GROUP_SIZE_K_SWITCH(group_size_k, GROUP_SIZE_K, [&]{
  //                     BLOCK_SIZE_M_LOOPS_SWITCH(block_size_m_loops , BLOCK_SIZE_M_LOOPS, [&]{
  //                     BLOCK_SIZE_N_LOOPS_SWITCH(block_size_n_loops , BLOCK_SIZE_N_LOOPS, [&]{
  //                       BLOCK_SIZE_K_LOOPS_SWITCH(block_size_k_loops , BLOCK_SIZE_K_LOOPS, [&]{
  //                         BOOL_SWITCH(use_atomic , USE_ATOMIC, [&]{
  //                   run_moe_wna16_gemm_block_wise<half, 8, TOPK, BLOCK_SIZE_M_, BLOCK_SIZE_N_, BLOCK_SIZE_K_, false, mul_topk_weight, GROUP_SIZE_N, GROUP_SIZE_K, BLOCK_SIZE_M_LOOPS, BLOCK_SIZE_N_LOOPS, BLOCK_SIZE_K_LOOPS, USE_ATOMIC,256>(
  //                     (const half*)input.data_ptr<at::Half>(),
  //                     // (const half*)d_input,
  //                     use_atomic ?(float*)output_fp32->data_ptr():(float*)output.data_ptr(),
  //                     // (float*)output_fp32->data_ptr(),
  //                     (const uint32_t*)b_qweight.data_ptr<int8_t>(),
  //                     // (const uint32_t*)d_w_test,
  //                     ( half_t*) d_w_out, /*for debug*/
  //                     // (const half*)b_scales.data_ptr<at::Half>(), 
  //                     (const float*)b_scales.data_ptr(),
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
  //                     });              
  //                     });
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
  // } else {
  // }
  // // half_t* tmp = reinterpret_cast<half_t*>(output.data_ptr());
  // // half_t* host_tmp = new half[1];  // 仅拷贝第一个值
  // // hipMemcpy(host_tmp, tmp, sizeof(half_t), hipMemcpyDeviceToHost);

  // // float first_value = __half2float(host_tmp[0]);
  // // delete[] host_tmp;

  // if (use_atomic){
  //   output.copy_(output_fp32->to(torch::kFloat16));
  // }
 
  return output;
  
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def(
      "moe_c_moe_gemm_marlin_w8a16("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor b_scale, "
      "Tensor? topk_weights, Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_pad, int top_k, int mode, int delta) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_w8a16",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_w8a16);

  m.def(
      "moe_c_moe_w8a16_gemm_block_wise("
      "Tensor input, Tensor! output, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int group_size_n, "
      "int group_size_k, int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, "
      "int BLOCK_SIZE_K, int bit) -> Tensor");
  m.impl(
      "moe_c_moe_w8a16_gemm_block_wise",
      torch::kCUDA,
      &moe_c_moe_w8a16_gemm_block_wise);

  m.def(
      "moe_c_moe_w8a16_gemm_awq("
      "Tensor input, Tensor! output, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, "
      "int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit) -> Tensor");
  m.impl(
      "moe_c_moe_w8a16_gemm_awq",
      torch::kCUDA,
      &moe_c_moe_w8a16_gemm_awq);
}
