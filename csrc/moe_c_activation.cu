#include <torch/extension.h>

#include "moe_wna16_utils.h"
#include "silu_mul_kernels.h"

void moe_c_activation_silu_and_mul(torch::Tensor& out,
                                   torch::Tensor& input,
                                   int64_t rows_per_block = 1,
                                   int64_t vec_size = 2) {
  moe_c::silu_and_mul(out, input, static_cast<int>(rows_per_block),
                      static_cast<int>(vec_size));
}

void moe_c_activation_gelu_and_mul(torch::Tensor& out,
                                   torch::Tensor& input,
                                   int64_t rows_per_block = 1,
                                   int64_t vec_size = 2) {
  moe_c::gelu_and_mul(out, input, static_cast<int>(rows_per_block),
                      static_cast<int>(vec_size));
}

void moe_c_activation_gelu_tanh_and_mul(torch::Tensor& out,
                                        torch::Tensor& input,
                                        int64_t rows_per_block = 1,
                                        int64_t vec_size = 2) {
  moe_c::gelu_tanh_and_mul(out, input, static_cast<int>(rows_per_block),
                           static_cast<int>(vec_size));
}

void moe_c_activation_gelu(torch::Tensor& out,
                           torch::Tensor& input,
                           int64_t rows_per_block = 1,
                           int64_t vec_size = 2) {
  moe_c::gelu(out, input, static_cast<int>(rows_per_block),
              static_cast<int>(vec_size));
}

void moe_c_activation_gelu_tanh(torch::Tensor& out,
                                torch::Tensor& input,
                                int64_t rows_per_block = 1,
                                int64_t vec_size = 2) {
  moe_c::gelu_tanh(out, input, static_cast<int>(rows_per_block),
                   static_cast<int>(vec_size));
}

void moe_c_activation_situ_glu(torch::Tensor& out,
                               torch::Tensor& input,
                               double beta1 = 4.0,
                               double beta2 = 25.0,
                               int64_t rows_per_block = 1,
                               int64_t vec_size = 2) {
  moe_c::situ_glu(out, input, static_cast<float>(beta1),
                  static_cast<float>(beta2),
                  static_cast<int>(rows_per_block),
                  static_cast<int>(vec_size));
}
