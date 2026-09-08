#include <unordered_map>
#include <functional>

template <typename T>
struct GemmParams_w8a16
{
    GemmParams_w8a16(const T *ptr_A,
                     const uint32_t *ptr_B0,
                     T *ptr_C,
                     T *ptr_B_scale,
                     const float *topk_weights,
                     const int32_t *sorted_token_ids,
                     const int32_t *expert_ids,
                     const int32_t num_tokens_post_pad,
                     const int32_t *num_tokens_post_pad_ptr,
                     uint32_t size_m,
                     uint32_t size_n,
                     uint32_t size_k,
                     uint32_t sorted_token_lens,
                     uint32_t top_k,
                     uint32_t delta,
                     bool is_marlin)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_B_scale(ptr_B_scale),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          delta(delta),
          is_marlin(is_marlin)
    {
    }

    const T *ptr_A;                         // input
    const uint32_t *ptr_B0;                 // weight
    T *ptr_C;                               // output
    T *ptr_B_scale;                         // weight scale
    const float *topk_weights;              // topk weights
    const int32_t *sorted_token_ids;        // sorted token ids
    const int32_t *expert_ids;              // expert ids
    const int32_t num_tokens_post_pad;      // num tokens after padding
    const int32_t *num_tokens_post_pad_ptr; // num tokens after padding
    uint32_t size_m;                        // input size m
    uint32_t size_n;                        // output size n
    uint32_t size_k;                        // input size k
    uint32_t sorted_token_lens;             // sorted token length
    uint32_t top_k;                         // top k
    uint32_t delta;                         // delta
    bool is_marlin;                         // 是否使用weight重排
};

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a16_first_stage_decode(const GemmParams_w8a16<T> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T>
void launch_moe_w8a16_second_stage_decode(const GemmParams_w8a16<T> &params);

template <typename scalar_t>
using KernelFunc_w8a16 = std::function<void(const GemmParams_w8a16<scalar_t> &)>;

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w8a16<scalar_t>> kernel_maps_gemm1_decode_w8a16 = {
    {0, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_first_stage_decode<16, 16, 512, 16, 16, 128, 2, scalar_t>(p); }},
    {1, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_first_stage_decode<16, 32, 512, 16, 32, 128, 2, scalar_t>(p); }},
    {3, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_first_stage_decode<16, 32, 256, 16, 16, 128, 2, scalar_t>(p); }},
    {4, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_first_stage_decode<16, 64, 256, 16, 32, 128, 2, scalar_t>(p); }},
    {5, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_first_stage_decode<16, 128, 256, 16, 64, 128, 2, scalar_t>(p); }},

};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w8a16<scalar_t>> kernel_maps_gemm2_decode_w8a16 = {
    {0, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_second_stage_decode<16, 16, 512, 16, 16, 128, 2, scalar_t>(p); }},
    {1, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_second_stage_decode<16, 32, 512, 16, 32, 128, 2, scalar_t>(p); }},
    {2, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_second_stage_decode<16, 64, 128, 16, 16, 128, 2, scalar_t>(p); }},
    {3, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_second_stage_decode<16, 32, 256, 16, 16, 128, 2, scalar_t>(p); }},
    {4, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_second_stage_decode<16, 64, 256, 16, 32, 128, 2, scalar_t>(p); }},
    {5, [](const GemmParams_w8a16<scalar_t> &p)
     { launch_moe_w8a16_second_stage_decode<16, 128, 256, 16, 64, 128, 2, scalar_t>(p); }},

};