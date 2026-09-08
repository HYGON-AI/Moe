// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace moe_c_detail {

template <typename... Args>
[[noreturn, gnu::noinline]] inline void hip_check_fatal(
    const char* file,
    size_t line,
    Args&&... args) {
  std::cerr << "[MOE] " << file << ":" << line << " ";
  (std::cerr << ... << std::forward<Args>(args)) << std::endl;
  std::abort();
}

}  // namespace moe_c_detail

#define HIP_CALL(call)                                                   \
  do {                                                                   \
    hipError_t err = call;                                               \
    if (err != hipSuccess) {                                             \
      moe_c_detail::hip_check_fatal(__FILE__,                            \
                                    __LINE__,                            \
                                    "fail to call " #call " ---> [HIP error](", \
                                    hipGetErrorString(err),              \
                                    ')');                                \
    }                                                                    \
  } while (0)

struct AiterAsmKernelArgs {
  void* args_ptr;
  void* arg_size_ptr;
  int gdx;
  int gdy;
  int gdz;
  int bdx;
  int bdy;
  int bdz;
  const hipStream_t stream;
};

class AiterAsmKernel {
 public:
  AiterAsmKernel(const char* name, const char* hsaco) {
    const char* asm_dir = std::getenv("AITER_ASM_DIR");
    if (asm_dir == nullptr) {
      moe_c_detail::hip_check_fatal(
          __FILE__, __LINE__, "AITER_ASM_DIR is required for ASM kernels");
    }

    const std::string path = std::string(asm_dir) + hsaco;
    std::cout << "[moe] hipModuleLoad: " << path << " GetFunction: " << name;
    HIP_CALL(hipModuleLoad(&module_, path.c_str()));
    HIP_CALL(hipModuleGetFunction(&kernel_func_, module_, name));
    std::cout << " Success" << std::endl;
  }

  ~AiterAsmKernel() {
    if (module_ != nullptr) {
      HIP_CALL(hipModuleUnload(module_));
    }
  }

  void launch_kernel(const AiterAsmKernelArgs& kargs) {
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      kargs.args_ptr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      kargs.arg_size_ptr,
                      HIP_LAUNCH_PARAM_END};

    HIP_CALL(hipModuleLaunchKernel(kernel_func_,
                                   kargs.gdx,
                                   kargs.gdy,
                                   kargs.gdz,
                                   kargs.bdx,
                                   kargs.bdy,
                                   kargs.bdz,
                                   0,
                                   kargs.stream,
                                   nullptr,
                                   reinterpret_cast<void**>(&config)));
  }

 private:
  hipModule_t module_{};
  hipFunction_t kernel_func_{};
};
