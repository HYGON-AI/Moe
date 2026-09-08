#pragma once

#include <functional>
#include <unordered_map>

#define MOE_THRESHOLD 128

namespace at
{
    namespace native
    {

        template <typename T>
        struct GemmParams3
        {
            GemmParams3(const T *ptr_A,
                        const T *ptr_B0,
                        T *ptr_C,
                        const float *topk_weights,
                        const int32_t *sorted_token_ids,
                        const int32_t *expert_ids,
                        const int32_t num_tokens_post_pad,
                        const int32_t *num_tokens_post_pad_ptr,
                        uint32_t size_m,
                        uint32_t size_n,
                        uint32_t size_k,
                        uint32_t size_kofb,
                        uint32_t sorted_token_lens,
                        uint32_t top_k,
                        uint32_t delta,
                        uint32_t expert_num)
                : ptr_A(ptr_A),
                  ptr_B0(ptr_B0),
                  ptr_C(ptr_C),
                  topk_weights(topk_weights),
                  sorted_token_ids(sorted_token_ids),
                  expert_ids(expert_ids),
                  num_tokens_post_pad(num_tokens_post_pad),
                  num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
                  size_m(size_m),
                  size_n(size_n),
                  size_k(size_k),
                  size_kofb(size_kofb),
                  sorted_token_lens(sorted_token_lens),
                  top_k(top_k),
                  delta(delta),
                  expert_num(expert_num) {}

            const T *ptr_A;
            const T *ptr_B0;
            T *ptr_C;
            const float *topk_weights;
            const int32_t *sorted_token_ids;
            const int32_t *expert_ids;
            const int32_t num_tokens_post_pad;
            const int32_t *num_tokens_post_pad_ptr;
            uint32_t size_m;
            uint32_t size_n;
            uint32_t size_k;
            uint32_t size_kofb;
            uint32_t sorted_token_lens;
            uint32_t top_k;
            uint32_t delta;
            uint32_t expert_num;
        };

        template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, typename T>
        void kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP(const GemmParams3<T> &params)
        {
            launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, splitk, T>(params);
        }

        template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int splitk, int N_LOOP, int STATIC_K, int STATIC_N, int STATIC_OUTPUT_N, int STATIC_LOOP_K, int CACHE_MODE, bool FUSE_NTILE2, typename T>
        void kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP(const GemmParams3<T> &params)
        {
            launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, splitk, N_LOOP, STATIC_K, STATIC_N, STATIC_OUTPUT_N, STATIC_LOOP_K, CACHE_MODE, FUSE_NTILE2, T>(params);
        }

        template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
        void kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN(const GemmParams3<T> &params)
        {
            launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
        }

        template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
        void kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP(const GemmParams3<T> &params)
        {
            launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
        }

        template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, typename T>
        void kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN(const GemmParams3<T> &params)
        {
            launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, T>(params);
        }

        template <typename T>
        using KernelFunc4 = std::function<void(const GemmParams3<T> &)>;

        template <typename T>
        static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm1_decode_marlin_w16a16()
        {
            return {
                {300, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 32, 16, 16, 16, 16, 1, T>},
                {301, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 32, 64, 16, 16, 16, 1, T>},
                {302, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 32, 128, 16, 16, 16, 1, T>},
                {303, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 32, 32, 16, 16, 32, 1, T>},
                {304, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 32, 128, 16, 16, 32, 1, T>},
                {305, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 32, 256, 16, 16, 32, 1, T>},
                {306, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 64, 16, 16, 16, 16, 1, T>},
                {307, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 128, 16, 16, 16, 16, 1, T>},
                {308, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 256, 16, 16, 16, 16, 1, T>},
                {309, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 64, 32, 16, 16, 32, 1, T>},
                {310, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 128, 32, 16, 16, 32, 1, T>},
                {311, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 256, 32, 16, 16, 32, 1, T>},
                {312, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 128, 64, 16, 16, 32, 1, T>},
                {313, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 64, 128, 16, 16, 32, 1, T>},
                {314, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 128, 32, 16, 16, 16, 1, T>},
                {315, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP<16, 64, 64, 16, 16, 16, 1, T>},
                {421, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP<16, 64, 128, 16, 16, 32, 1, 1, 1232, 0, 0, 1248, 0, false, T>},
                {367, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP<16, 128, 32, 16, 16, 32, 1, 2, 1232, 0, 0, 1248, 3, true, T>},
                {369, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_UP_TOPK1_NLOOP<16, 128, 32, 16, 16, 32, 1, 1, 1280, 0, 0, 1280, 3, false, T>},
            };
        }

        static auto kernel_maps_gemm1_decode_marlin_w16a16_half = make_kernel_maps_gemm1_decode_marlin_w16a16<half>();
        static auto kernel_maps_gemm1_decode_marlin_w16a16_bhalf_t = make_kernel_maps_gemm1_decode_marlin_w16a16<bhalf_t>();

        template <typename T>
        static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm2_decode_marlin_w16a16()
        {
            return {
                {300, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 32, 16, 16, 16, 16, T>},
                {301, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 32, 64, 16, 16, 16, T>},
                {302, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 32, 128, 16, 16, 16, T>},
                {303, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 32, 32, 16, 16, 32, T>},
                {304, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 32, 128, 16, 16, 32, T>},
                {305, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 32, 256, 16, 16, 32, T>},
                {306, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 64, 16, 16, 16, 16, T>},
                {307, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 128, 16, 16, 16, 16, T>},
                {308, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 256, 16, 16, 16, 16, T>},
                {309, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 64, 32, 16, 16, 32, T>},
                {310, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 128, 32, 16, 16, 32, T>},
                {311, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 256, 32, 16, 16, 32, T>},
                {312, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 128, 64, 16, 16, 32, T>},
                {313, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 64, 128, 16, 16, 32, T>},
                {314, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 128, 32, 16, 16, 16, T>},
                {315, &kernel_launch_MOE_W16A16_MARLIN_HIP_NN_DECODE_DOWN<16, 64, 64, 16, 16, 16, T>},
            };
        }

        static auto kernel_maps_gemm2_decode_marlin_w16a16_half = make_kernel_maps_gemm2_decode_marlin_w16a16<half>();
        static auto kernel_maps_gemm2_decode_marlin_w16a16_bhalf_t = make_kernel_maps_gemm2_decode_marlin_w16a16<bhalf_t>();

        template <typename T>
        static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm1_prefill_marlin_w16a16()
        {
            return {
                {10, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 32, 64, T>},
                {11, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 32, 64, T>},
                {12, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 32, 64, T>},
                {13, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {14, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {15, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {16, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {17, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {18, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {21, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {32, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 32, 64, T>},
                {33, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 32, 64, T>},
                {34, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 32, 64, T>},
                {35, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {36, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {37, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {38, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {39, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {40, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {41, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {42, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {44, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {46, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {103, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 32, 64, T>},
                {104, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 128, 32, 32, 64, T>},
                {106, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 256, 128, 16, 64, 64, T>},
                {114, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 256, 64, 64, 64, 32, T>},
                {115, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 64, 128, 16, 32, 64, T>},
                {117, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 64, 128, 32, 32, 64, T>},
                {120, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 128, 16, 32, 64, T>},
                {121, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 128, 32, 32, 64, T>},
                {123, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 64, 128, 16, 32, 64, T>},
                {128, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 32, 128, 16, 32, 64, T>},
                {129, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 64, 128, 16, 64, 64, T>},
                {130, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 64, 128, 16, 64, 64, T>},
                {132, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 32, 128, 16, 32, 64, T>},
                {133, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 64, 128, 16, 64, 64, T>},
                {136, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 32, 128, 16, 32, 64, T>},
                {141, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 128, 32, 32, 32, T>},
                {143, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 256, 128, 32, 64, 32, T>},
                {146, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 256, 128, 64, 32, 32, T>},
                {156, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 128, 128, 16, 64, 64, T>},
                {168, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 128, 64, 32, 32, 32, T>},
                {175, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 64, 64, 16, 32, 32, T>},
                {177, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 64, 64, 32, 32, 32, T>},
                {181, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 128, 64, 32, 32, 32, T>},
                {188, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 32, 64, 16, 32, 32, T>},
                {189, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 64, 64, 16, 64, 32, T>},
                {190, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<16, 64, 64, 16, 64, 32, T>},
                {192, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 32, 64, 16, 32, 32, T>},
                {193, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<32, 64, 64, 16, 64, 32, T>},
                {195, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_UP<64, 32, 64, 16, 32, 32, T>},
            };
        }

        static auto kernel_maps_gemm1_prefill_marlin_w16a16_half = make_kernel_maps_gemm1_prefill_marlin_w16a16<half>();
        static auto kernel_maps_gemm1_prefill_marlin_w16a16_bhalf_t = make_kernel_maps_gemm1_prefill_marlin_w16a16<bhalf_t>();

        template <typename T>
        static std::unordered_map<int, KernelFunc4<T>> make_kernel_maps_gemm2_prefill_marlin_w16a16()
        {
            return {
                {10, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 256, 32, 16, 64, 32, T>},
                {11, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 256, 32, 16, 32, 32, T>},
                {12, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 128, 32, 16, 32, 32, T>},
                {13, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 256, 32, 32, 32, 32, T>},
                {14, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 256, 32, 32, 64, 32, T>},
                {15, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 256, 32, 16, 64, 32, T>},
                {16, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 128, 32, 16, 32, 32, T>},
                {17, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 256, 32, 64, 64, 32, T>},
                {18, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 256, 32, 64, 32, 32, T>},
                {21, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 128, 32, 64, 32, 32, T>},
                {32, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 256, 16, 16, 64, 16, T>},
                {33, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 256, 16, 16, 32, 16, T>},
                {34, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 128, 16, 16, 32, 16, T>},
                {35, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 256, 16, 32, 32, 16, T>},
                {36, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 256, 16, 32, 64, 16, T>},
                {37, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 256, 16, 16, 64, 16, T>},
                {38, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 128, 16, 32, 32, 16, T>},
                {39, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 128, 16, 16, 32, 16, T>},
                {40, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 256, 16, 64, 64, 16, T>},
                {41, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 256, 16, 64, 32, 16, T>},
                {42, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 256, 16, 32, 64, 16, T>},
                {44, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 128, 16, 64, 32, 16, T>},
                {46, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 128, 16, 16, 64, 16, T>},
                {103, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 128, 128, 16, 32, 64, T>},
                {104, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 128, 128, 32, 32, 64, T>},
                {106, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 256, 128, 16, 64, 64, T>},
                {114, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 256, 32, 64, 64, 32, T>},
                {115, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 64, 128, 16, 32, 64, T>},
                {117, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 64, 128, 32, 32, 64, T>},
                {120, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 128, 128, 16, 32, 64, T>},
                {121, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 128, 128, 32, 32, 64, T>},
                {123, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 64, 128, 16, 32, 64, T>},
                {128, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 32, 128, 16, 32, 64, T>},
                {129, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 64, 128, 16, 64, 64, T>},
                {130, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 64, 128, 16, 64, 64, T>},
                {132, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 32, 128, 16, 32, 64, T>},
                {133, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 64, 128, 16, 64, 64, T>},
                {136, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 32, 128, 16, 32, 64, T>},
                {141, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 128, 128, 32, 32, 32, T>},
                {143, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 256, 128, 32, 64, 32, T>},
                {146, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 256, 128, 64, 32, 32, T>},
                {156, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 128, 128, 16, 64, 64, T>},
                {168, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 128, 64, 32, 32, 32, T>},
                {175, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 64, 64, 16, 32, 32, T>},
                {177, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 64, 64, 32, 32, 32, T>},
                {181, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 128, 64, 32, 32, 32, T>},
                {188, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 32, 64, 16, 32, 32, T>},
                {189, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 64, 64, 16, 64, 32, T>},
                {190, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<16, 64, 64, 16, 64, 32, T>},
                {192, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 32, 64, 16, 32, 32, T>},
                {193, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<32, 64, 64, 16, 64, 32, T>},
                {195, &kernel_launch_MOE_W16A16_MARLIN_HIP_NT_PREFILL_DOWN<64, 32, 64, 16, 32, 32, T>},
            };
        }

        static auto kernel_maps_gemm2_prefill_marlin_w16a16_half = make_kernel_maps_gemm2_prefill_marlin_w16a16<half>();
        static auto kernel_maps_gemm2_prefill_marlin_w16a16_bhalf_t = make_kernel_maps_gemm2_prefill_marlin_w16a16<bhalf_t>();

    } // namespace native
} // namespace at
