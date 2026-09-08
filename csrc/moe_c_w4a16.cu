#include "moe_c_common.h"

torch::Tensor moe_c_moe_gemm_marlin_w4a16(torch::Tensor input,
  torch::Tensor b_qweight,
  torch::Tensor output,
  torch::Tensor b_scale,
  torch::Tensor b_zeros,
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
    const int size_n = b_scale.size(1);
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

    const float* topk_weights_ptr; // 第一阶段这里为null
    if (topk_weights.has_value()){
      topk_weights_ptr = (const float*)topk_weights.value().data_ptr();
      first_stage = false;
    }

    int num_pad = 0;

    if (input.scalar_type() == at::ScalarType::Half){
      if(first_stage){
     
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        GemmParams_w4a16<half> params_in(
          (half*)input.data_ptr<at::Half>(), 
          (uint32_t*)b_qweight.data_ptr<uint32_t>(), 
          (half*)output_alias.data_ptr<at::Half>(),
          reinterpret_cast<uint32_t*>(b_zeros.data_ptr<uint8_t>()), 
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
          auto it = kernel_maps_gemm1_decode_w4a16<half>.find(mode);
          if (it != kernel_maps_gemm1_decode_w4a16<half>.end() ) {
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

              std::ofstream ofs("./w4a16_kernel_1_timecost", std::ios::app); // 追加写入
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
        GemmParams_w4a16<half> params_in(
          (half*)input.data_ptr<at::Half>(), 
          (uint32_t*)b_qweight.data_ptr<uint32_t>(), 
          (half*)output_alias.data_ptr<at::Half>(),
          reinterpret_cast<uint32_t*>(b_zeros.data_ptr<uint8_t>()), 
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
          
          auto it = kernel_maps_gemm2_decode_w4a16<half>.find(mode);
          if (it != kernel_maps_gemm2_decode_w4a16<half>.end() ) {
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

            std::ofstream ofs("./w4a16_kernel_2_timecost", std::ios::app); // 追加写入
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
      if(first_stage){
     
        int64_t EM = sorted_token_ids.size(0); // 一维线性化的token id
        GemmParams_w4a16<__hip_bfloat16> params_in(
          (__hip_bfloat16*)input.data_ptr<at::BFloat16>(), 
          (uint32_t*)b_qweight.data_ptr<uint32_t>(), 
          (__hip_bfloat16*)output_alias.data_ptr<at::BFloat16>(),
          reinterpret_cast<uint32_t*>(b_zeros.data_ptr<uint8_t>()), 
          (__hip_bfloat16*)b_scale.data_ptr<at::BFloat16>(),  
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
          auto it = kernel_maps_gemm1_decode_w4a16<__hip_bfloat16>.find(mode);
          if (it != kernel_maps_gemm1_decode_w4a16<__hip_bfloat16>.end() ) {
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

              std::ofstream ofs("./w4a16_kernel_1_timecost", std::ios::app); // 追加写入
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
        GemmParams_w4a16<__hip_bfloat16> params_in(
          (__hip_bfloat16*)input.data_ptr<at::BFloat16>(), 
          (uint32_t*)b_qweight.data_ptr<uint32_t>(), 
          (__hip_bfloat16*)output_alias.data_ptr<at::BFloat16>(),
          reinterpret_cast<uint32_t*>(b_zeros.data_ptr<uint8_t>()), 
          (__hip_bfloat16*)b_scale.data_ptr<at::BFloat16>(),  
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
          
          auto it = kernel_maps_gemm2_decode_w4a16<__hip_bfloat16>.find(mode);
          if (it != kernel_maps_gemm2_decode_w4a16<__hip_bfloat16>.end() ) {
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

            std::ofstream ofs("./w4a16_kernel_2_timecost", std::ios::app); // 追加写入
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
    } else {
      TORCH_CHECK(false, "moe_w8a8_gemm only supports int8");
    }

    return output;
}

TORCH_LIBRARY_FRAGMENT(moe_wna16, m) {
  m.def(
      "moe_c_moe_gemm_marlin_w4a16("
      "Tensor input, Tensor b_qweight, Tensor output, Tensor b_scale, "
      "Tensor b_zeros, Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, int top_k, int mode, "
      "int delta) -> Tensor");
  m.impl(
      "moe_c_moe_gemm_marlin_w4a16",
      torch::kCUDA,
      &moe_c_moe_gemm_marlin_w4a16);
}
