#ifndef MOE_WFP4A8_UTILS_HIP_H
#define MOE_WFP4A8_UTILS_HIP_H

#include <torch/all.h>
#ifdef USE_ROCM
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <hip/hip_runtime.h>

#include "moe_wna16_utils.h"
#include "intrinsic_2.h"
#include "intrinsic.h"
#include "numeric_types.h"

constexpr int WFP4A8_FP4X2_LUT_BYTES = 256 * sizeof(uint32_t);

#ifndef WFP4A8_SKIP_DEQUANT
#define WFP4A8_SKIP_DEQUANT 0
#endif

#ifndef WFP4A8_DEQUANT_PERM
#define WFP4A8_DEQUANT_PERM 0
#endif

#ifndef WFP4A8_BLOAD_PIPELINE
#define WFP4A8_BLOAD_PIPELINE 0
#endif

#ifndef WFP4A8_SKIP_GROUP_SCALE
#define WFP4A8_SKIP_GROUP_SCALE 0
#endif

#if WFP4A8_BLOAD_PIPELINE
#define WFP4A8_BPIPE_BARRIER() \
    do                         \
    {                          \
    } while (0)
#else
#define WFP4A8_BPIPE_BARRIER() __builtin_amdgcn_sched_barrier(0)
#endif

__device__ __forceinline__ uint8_t wfp4a8_fp4_e2m1_to_fp8_e4m3(uint32_t code)
{
    constexpr uint8_t lut[16] = {
        0x00,
        0x30,
        0x38,
        0x3c,
        0x40,
        0x44,
        0x48,
        0x4c,
        0x80,
        0xb0,
        0xb8,
        0xbc,
        0xc0,
        0xc4,
        0xc8,
        0xcc,
    };
    return lut[code & 0xf];
}

__device__ __forceinline__ uint32_t wfp4a8_make_fp4x2_to_fp8x2(uint32_t packed_byte)
{
    const uint32_t idx = packed_byte & 0xff;
    const uint32_t low = wfp4a8_fp4_e2m1_to_fp8_e4m3(idx);
    const uint32_t high = wfp4a8_fp4_e2m1_to_fp8_e4m3(idx >> 4);
    return low | (high << 8);
}

__device__ __forceinline__ void wfp4a8_init_fp4x2_lut(uint32_t *lut_lds)
{
    for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x)
    {
        lut_lds[i] = wfp4a8_make_fp4x2_to_fp8x2(i);
    }
    __syncthreads();
}

__device__ __forceinline__ uint16_t wfp4a8_fp4x2_to_fp8x2(uint32_t packed_byte)
{
    constexpr uint16_t lut[256] = {
        0x0000, 0x0030, 0x0038, 0x003c, 0x0040, 0x0044, 0x0048, 0x004c,
        0x0080, 0x00b0, 0x00b8, 0x00bc, 0x00c0, 0x00c4, 0x00c8, 0x00cc,
        0x3000, 0x3030, 0x3038, 0x303c, 0x3040, 0x3044, 0x3048, 0x304c,
        0x3080, 0x30b0, 0x30b8, 0x30bc, 0x30c0, 0x30c4, 0x30c8, 0x30cc,
        0x3800, 0x3830, 0x3838, 0x383c, 0x3840, 0x3844, 0x3848, 0x384c,
        0x3880, 0x38b0, 0x38b8, 0x38bc, 0x38c0, 0x38c4, 0x38c8, 0x38cc,
        0x3c00, 0x3c30, 0x3c38, 0x3c3c, 0x3c40, 0x3c44, 0x3c48, 0x3c4c,
        0x3c80, 0x3cb0, 0x3cb8, 0x3cbc, 0x3cc0, 0x3cc4, 0x3cc8, 0x3ccc,
        0x4000, 0x4030, 0x4038, 0x403c, 0x4040, 0x4044, 0x4048, 0x404c,
        0x4080, 0x40b0, 0x40b8, 0x40bc, 0x40c0, 0x40c4, 0x40c8, 0x40cc,
        0x4400, 0x4430, 0x4438, 0x443c, 0x4440, 0x4444, 0x4448, 0x444c,
        0x4480, 0x44b0, 0x44b8, 0x44bc, 0x44c0, 0x44c4, 0x44c8, 0x44cc,
        0x4800, 0x4830, 0x4838, 0x483c, 0x4840, 0x4844, 0x4848, 0x484c,
        0x4880, 0x48b0, 0x48b8, 0x48bc, 0x48c0, 0x48c4, 0x48c8, 0x48cc,
        0x4c00, 0x4c30, 0x4c38, 0x4c3c, 0x4c40, 0x4c44, 0x4c48, 0x4c4c,
        0x4c80, 0x4cb0, 0x4cb8, 0x4cbc, 0x4cc0, 0x4cc4, 0x4cc8, 0x4ccc,
        0x8000, 0x8030, 0x8038, 0x803c, 0x8040, 0x8044, 0x8048, 0x804c,
        0x8080, 0x80b0, 0x80b8, 0x80bc, 0x80c0, 0x80c4, 0x80c8, 0x80cc,
        0xb000, 0xb030, 0xb038, 0xb03c, 0xb040, 0xb044, 0xb048, 0xb04c,
        0xb080, 0xb0b0, 0xb0b8, 0xb0bc, 0xb0c0, 0xb0c4, 0xb0c8, 0xb0cc,
        0xb800, 0xb830, 0xb838, 0xb83c, 0xb840, 0xb844, 0xb848, 0xb84c,
        0xb880, 0xb8b0, 0xb8b8, 0xb8bc, 0xb8c0, 0xb8c4, 0xb8c8, 0xb8cc,
        0xbc00, 0xbc30, 0xbc38, 0xbc3c, 0xbc40, 0xbc44, 0xbc48, 0xbc4c,
        0xbc80, 0xbcb0, 0xbcb8, 0xbcbc, 0xbcc0, 0xbcc4, 0xbcc8, 0xbccc,
        0xc000, 0xc030, 0xc038, 0xc03c, 0xc040, 0xc044, 0xc048, 0xc04c,
        0xc080, 0xc0b0, 0xc0b8, 0xc0bc, 0xc0c0, 0xc0c4, 0xc0c8, 0xc0cc,
        0xc400, 0xc430, 0xc438, 0xc43c, 0xc440, 0xc444, 0xc448, 0xc44c,
        0xc480, 0xc4b0, 0xc4b8, 0xc4bc, 0xc4c0, 0xc4c4, 0xc4c8, 0xc4cc,
        0xc800, 0xc830, 0xc838, 0xc83c, 0xc840, 0xc844, 0xc848, 0xc84c,
        0xc880, 0xc8b0, 0xc8b8, 0xc8bc, 0xc8c0, 0xc8c4, 0xc8c8, 0xc8cc,
        0xcc00, 0xcc30, 0xcc38, 0xcc3c, 0xcc40, 0xcc44, 0xcc48, 0xcc4c,
        0xcc80, 0xccb0, 0xccb8, 0xccbc, 0xccc0, 0xccc4, 0xccc8, 0xcccc};
    return lut[packed_byte & 0xff];
}

__device__ __forceinline__ uint32_t wfp4a8_fp4x4_to_fp8_perm(uint32_t packed_x4)
{
    constexpr uint32_t lut0 = 0x3c383000u;
    constexpr uint32_t lut1 = 0x4c484440u;
    const uint32_t mag = packed_x4 & 0x7777u;
    const uint32_t sel =
        (mag & 0x0007u) |
        ((mag & 0x0070u) << 4) |
        ((mag & 0x0700u) << 8) |
        ((mag & 0x7000u) << 12);
    const uint32_t sign =
        ((packed_x4 & 0x0008u) << 4) |
        ((packed_x4 & 0x0080u) << 8) |
        ((packed_x4 & 0x0800u) << 12) |
        ((packed_x4 & 0x8000u) << 16);
    return __builtin_amdgcn_perm(lut1, lut0, sel) | sign;
}

__device__ __forceinline__ uint32_t wfp4a8_pack_high_fp4x4_to_fp8(uint32_t packed)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#elif WFP4A8_DEQUANT_PERM
    return wfp4a8_fp4x4_to_fp8_perm(packed);
#else
    const uint32_t out0 = wfp4a8_fp4x2_to_fp8x2(packed);
    const uint32_t out1 = wfp4a8_fp4x2_to_fp8x2(packed >> 8);
    return out0 | (out1 << 16);
#endif
}

__device__ __forceinline__ uint32_t wfp4a8_pack_low_fp4x4_to_fp8(uint32_t packed)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#elif WFP4A8_DEQUANT_PERM
    return wfp4a8_fp4x4_to_fp8_perm(packed >> 16);
#else
    const uint32_t out0 = wfp4a8_fp4x2_to_fp8x2(packed >> 16);
    const uint32_t out1 = wfp4a8_fp4x2_to_fp8x2(packed >> 24);
    return out0 | (out1 << 16);
#endif
}

__device__ __forceinline__ float wfp4a8_e8m0_to_float(uint8_t scale)
{
    union
    {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = static_cast<uint32_t>(scale) << 23;
    return cvt.f;
}

template <bool FastScaleAddr, int SIZE_K>
__device__ __forceinline__ float load_gs(
    const uint8_t *__restrict__ ptr,
    int logical_n,
    int group_k,
    int stride_n,
    int stride_k)
{
    if constexpr (FastScaleAddr)
    {
        return wfp4a8_e8m0_to_float(ptr[logical_n * (SIZE_K / 32) + group_k]);
    }
    else
    {
        return wfp4a8_e8m0_to_float(ptr[logical_n * stride_n + group_k * stride_k]);
    }
}

__device__ __forceinline__ float load_gs_at(const uint8_t *__restrict__ ptr, int offset)
{
    return wfp4a8_e8m0_to_float(ptr[offset]);
}

__device__ __forceinline__ int load_arow(
    const int32_t *__restrict__ ids,
    int offset,
    int len,
    int top_k,
    int max_m)
{
    const int32_t sid = ids[std::min(offset, len - 1)];
    return std::min(sid / top_k, max_m - 1);
}

__device__ __forceinline__ float load_as(
    const float *__restrict__ ptr,
    int row,
    int group_k,
    int stride_m,
    int stride_k)
{
    return ptr[row * stride_m + group_k * stride_k];
}

template <int WARP_M, int WARP_K, int MFMA_M, int MFMA_K, int BLOCK_M>
__device__ __forceinline__ void load_as_tile(
    float (&as)[WARP_M / MFMA_M][4][WARP_K / MFMA_K],
    const float *__restrict__ ptr,
    const int32_t *ids,
    int group_k0,
    int bidx,
    int col_id,
    int len,
    int top_k,
    int max_m,
    int stride_m,
    int stride_k)
{
#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int reg_id = 0; reg_id < 4; reg_id++)
        {
            const int a_row = load_arow(
                ids,
                bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                len,
                top_k,
                max_m);
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
            {
                as[m_tile][reg_id][k_tile] = load_as(ptr, a_row, group_k0 + k_tile, stride_m, stride_k);
            }
        }
    }
}

template <
    int BLOCK_M,
    int BLOCK_N,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int MFMA_M,
    int MFMA_K,
    int N_LOOP_NUM,
    bool FastScaleAddr,
    int SIZE_K>
__device__ __forceinline__ void load_scale_slot(
    uint8_t *scale_slot,
    const float *__restrict__ input_scale_ptr,
    const uint8_t *__restrict__ weight_scale_u8_ptr,
    const int32_t *__restrict__ sorted_token_ids_offset,
    int group_k0,
    int bidx,
    int warp_n_id,
    int row_id,
    int col_id,
    int sorted_token_lens,
    int top_k,
    int max_n_len_offset,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_n,
    int scale_B_stride_k)
{
    constexpr int K_TILE_NUM = WARP_K / MFMA_K;
    constexpr int A_SCALE_LDS_FLOATS = BLOCK_M * K_TILE_NUM;
    float *a_scale_lds = reinterpret_cast<float *>(scale_slot);
    uint8_t *b_scale_u8_lds = scale_slot + A_SCALE_LDS_FLOATS * sizeof(float);

    if (warp_n_id == 0 && row_id == 0)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int reg_id = 0; reg_id < 4; reg_id++)
            {
                const int a_row = load_arow(
                    sorted_token_ids_offset,
                    bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                    sorted_token_lens,
                    top_k,
                    max_n_len_offset);
                const int m_local = m_tile * MFMA_M + reg_id * 4 + col_id;
#pragma unroll
                for (int k_tile = 0; k_tile < K_TILE_NUM; k_tile++)
                {
                    a_scale_lds[m_local * K_TILE_NUM + k_tile] =
                        load_as(input_scale_ptr, a_row, group_k0 + k_tile, scale_A_stride_m, scale_A_stride_k);
                }
            }
        }
    }

    if (col_id == 0)
    {
#pragma unroll
        for (int scale_loop = 0; scale_loop < N_LOOP_NUM; scale_loop++)
        {
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
                const int local_n_base = warp_n_id * WARP_N + n_tile * 32 + row_id * 2;
                const int logical_n_base = BLOCK_N * scale_loop + local_n_base;
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
                    const int local_n = local_n_base + it;
                    const int logical_n = logical_n_base + it;
                    const int scale_addr_base =
                        FastScaleAddr
                            ? logical_n * (SIZE_K / 32) + group_k0
                            : logical_n * scale_B_stride_n + group_k0 * scale_B_stride_k;
#pragma unroll
                    for (int k_tile = 0; k_tile < K_TILE_NUM; k_tile++)
                    {
                        const int scale_addr = scale_addr_base + k_tile * (FastScaleAddr ? 1 : scale_B_stride_k);
                        b_scale_u8_lds[(scale_loop * BLOCK_N + local_n) * K_TILE_NUM + k_tile] =
                            weight_scale_u8_ptr[scale_addr];
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ float wfp4a8_fp4_e2m1_to_float(uint32_t code)
{
    constexpr float lut[16] = {
        0.0f,
        0.5f,
        1.0f,
        1.5f,
        2.0f,
        3.0f,
        4.0f,
        6.0f,
        -0.0f,
        -0.5f,
        -1.0f,
        -1.5f,
        -2.0f,
        -3.0f,
        -4.0f,
        -6.0f,
    };
    return lut[code & 0xf];
}

__device__ __forceinline__ uint8_t wfp4a8_fp4_e2m1_e8m0_to_fp8(uint32_t code, uint8_t scale)
{
    const uint32_t mag = code & 0x7;
    const uint32_t sign = (code & 0x8) ? 0x80 : 0;
    if (mag == 0)
    {
        return static_cast<uint8_t>(sign);
    }

    const int exp_delta = static_cast<int>(mag >> 1) - 1;
    const bool half_mant = (mag & 1) && mag != 1;
    const int exp = static_cast<int>(scale) - 120 + exp_delta;

    if (exp >= 16)
    {
        return static_cast<uint8_t>(sign | 0x7e);
    }
    if (exp >= 1)
    {
        return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | (half_mant ? 0x4 : 0));
    }

    uint32_t mant = 0;
    if (exp == 0)
    {
        mant = half_mant ? 6 : 4;
    }
    else if (exp == -1)
    {
        mant = half_mant ? 3 : 2;
    }
    else if (exp == -2)
    {
        mant = half_mant ? 2 : 1;
    }
    else if (exp == -3)
    {
        mant = half_mant ? 1 : 0;
    }
    return static_cast<uint8_t>(sign | mant);
}

__device__ __forceinline__ uint8_t wfp4a8_scale_fp8_pow2(uint32_t base, uint8_t scale)
{
    const uint32_t sign = base & 0x80;
    const uint32_t exp_base = base & 0x78;
    const uint32_t mant_base = base & 0x7;
    if (exp_base == 0)
    {
        return static_cast<uint8_t>(sign);
    }

    const int exp = static_cast<int>(exp_base >> 3) + static_cast<int>(scale) - 127;
    if (exp >= 15)
    {
        return static_cast<uint8_t>(sign | 0x7e);
    }
    if (exp >= 1)
    {
        return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | mant_base);
    }

    const uint32_t sig = 8 | mant_base;
    const int shift = 1 - exp;
    if (shift >= 5)
    {
        return static_cast<uint8_t>(sign);
    }
    const uint32_t mant = sig >> shift;
    return static_cast<uint8_t>(sign | mant);
}

__device__ __forceinline__ uint32_t wfp4a8_scale_fp8x4_pow2(uint32_t base, uint8_t scale)
{
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        out |= static_cast<uint32_t>(wfp4a8_scale_fp8_pow2((base >> (i * 8)) & 0xff, scale)) << (i * 8);
    }
    return out;
}

__device__ __forceinline__ uint32_t wfp4a8_pack_high_fp4x4_e8m0_to_fp8(uint32_t packed, uint8_t scale)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#else
    return wfp4a8_scale_fp8x4_pow2(wfp4a8_pack_high_fp4x4_to_fp8(packed), scale);
#endif
}

__device__ __forceinline__ uint32_t wfp4a8_pack_low_fp4x4_e8m0_to_fp8(uint32_t packed, uint8_t scale)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#else
    return wfp4a8_scale_fp8x4_pow2(wfp4a8_pack_low_fp4x4_to_fp8(packed), scale);
#endif
}

__device__ __forceinline__ void wfp4a8_pack_raw_fp4x8_to_fp8(uint32_t packed, uint32_t &out0, uint32_t &out1)
{
#if WFP4A8_SKIP_DEQUANT
    out0 = 0x30303030u;
    out1 = 0x30303030u;
#else
    out0 = 0;
    out1 = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        const uint32_t v0 = wfp4a8_fp4_e2m1_to_fp8_e4m3((packed >> (i * 4)) & 0xf);
        const uint32_t v1 = wfp4a8_fp4_e2m1_to_fp8_e4m3((packed >> (16 + i * 4)) & 0xf);
        if (i < 2)
        {
            out0 |= v0 << (i * 16);
            out0 |= v1 << (i * 16 + 8);
        }
        else
        {
            out1 |= v0 << ((i - 2) * 16);
            out1 |= v1 << ((i - 2) * 16 + 8);
        }
    }
#endif
}

__device__ __forceinline__ void wfp4a8_pack_raw_fp4x8_e8m0_to_fp8(
    uint32_t packed, uint8_t scale, uint32_t &out0, uint32_t &out1)
{
#if WFP4A8_SKIP_DEQUANT
    out0 = 0x30303030u;
    out1 = 0x30303030u;
#else
    out0 = 0;
    out1 = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        const uint32_t v0 = wfp4a8_fp4_e2m1_e8m0_to_fp8((packed >> (i * 4)) & 0xf, scale);
        const uint32_t v1 = wfp4a8_fp4_e2m1_e8m0_to_fp8((packed >> (16 + i * 4)) & 0xf, scale);
        if (i < 2)
        {
            out0 |= v0 << (i * 16);
            out0 |= v1 << (i * 16 + 8);
        }
        else
        {
            out1 |= v0 << ((i - 2) * 16);
            out1 |= v1 << ((i - 2) * 16 + 8);
        }
    }
#endif
}

__device__ __forceinline__ uint32_t wfp4a8_fp4x2_to_fp8x2_lds(
    const uint32_t *lut_lds,
    uint32_t packed_byte)
{
    const uint32_t idx = packed_byte & 0xff;
    return lut_lds[idx];
}

__device__ __forceinline__ uint32_t wfp4a8_pack_high_fp4x4_to_fp8(
    const uint32_t *lut_lds,
    uint32_t packed)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#elif WFP4A8_DEQUANT_PERM
    return wfp4a8_fp4x4_to_fp8_perm(packed);
#else
    const uint32_t out0 = wfp4a8_fp4x2_to_fp8x2_lds(lut_lds, packed);
    const uint32_t out1 = wfp4a8_fp4x2_to_fp8x2_lds(lut_lds, packed >> 8);
    return out0 | (out1 << 16);
#endif
}

__device__ __forceinline__ uint32_t wfp4a8_pack_low_fp4x4_to_fp8(
    const uint32_t *lut_lds,
    uint32_t packed)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#elif WFP4A8_DEQUANT_PERM
    return wfp4a8_fp4x4_to_fp8_perm(packed >> 16);
#else
    const uint32_t out0 = wfp4a8_fp4x2_to_fp8x2_lds(lut_lds, packed >> 16);
    const uint32_t out1 = wfp4a8_fp4x2_to_fp8x2_lds(lut_lds, packed >> 24);
    return out0 | (out1 << 16);
#endif
}

__device__ __forceinline__ uint32_t wfp4a8_pack_high_fp4x4_e8m0_to_fp8(
    const uint32_t *lut_lds,
    uint32_t packed,
    uint8_t scale)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#else
    return wfp4a8_scale_fp8x4_pow2(wfp4a8_pack_high_fp4x4_to_fp8(lut_lds, packed), scale);
#endif
}

__device__ __forceinline__ uint32_t wfp4a8_pack_low_fp4x4_e8m0_to_fp8(
    const uint32_t *lut_lds,
    uint32_t packed,
    uint8_t scale)
{
#if WFP4A8_SKIP_DEQUANT
    return 0x30303030u;
#else
    return wfp4a8_scale_fp8x4_pow2(wfp4a8_pack_low_fp4x4_to_fp8(lut_lds, packed), scale);
#endif
}

template <
    int WARP_NUM,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int STAGE,
    int SIZE_K,
    typename Element,
    bool GroupwisePost = false,
    bool AGroupScale = true>
__forceinline__ __device__ void wfp4a8_mmac_tail64(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int size_n,
    const int *g_row_A,
    const int *g_row_B,
    int n_loop_num,
    const uint8_t *weight_scale_u8_ptr = nullptr,
    int scale_B_stride_n = 0,
    int scale_B_stride_k = 0,
    const uint32_t *fp4x2_lut_lds = nullptr,
    const float *input_scale_ptr = nullptr,
    int scale_A_stride_m = 0,
    int scale_A_stride_k = 0,
    const int32_t *sorted_token_ids_offset = nullptr,
    int sorted_token_lens = 0,
    int top_k = 1,
    int max_n_len_offset = 0,
    int bidx = 0)
{
    if constexpr ((SIZE_K % 128 == 64) && (WARP_K == 64))
    {
        __syncthreads();

        constexpr int MFMA_M = 16;
        constexpr int MFMA_N = 16;
        constexpr int MFMA_K = 32;
        constexpr int READ_K = 64;
        constexpr int warp_k_num = BLOCK_K / WARP_K;

        const int lane_id = threadIdx.x & 63;
        const int row_id = lane_id % 16;
        const int col_id = lane_id / 16;
        const int warp_k_id = warp_id % warp_k_num;
        const int warp_n_id = warp_id / warp_k_num;
        const int A_index = warp_id;
        const int k_start = warp_k_id * WARP_K + (SIZE_K / 128) * 128;
        const int k_start_b = warp_k_id * (WARP_K / 2) * size_n + (SIZE_K / 128) * (128 / 2) * size_n;
        auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                inline_buffer_load_dword_lds(
                    A_lds,
                    g_input,
                    (m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                    (k_tile * READ_K + k_start) / 4,
                    g_row_A[m_tile] / 4);
            }
        }

        vmcnt_only_wait(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                int s_index = m_tile * 16 * WARP_K;
                if constexpr (GroupwisePost)
                {
                    A_reg[m_tile][0].int8t_array[0] =
                        *(vec<Element, 8> *)(&A_lds[s_index + row_id * 64 + col_id * 8]);
                    A_reg[m_tile][0].int8t_array[1] =
                        *(vec<Element, 8> *)(&A_lds[s_index + row_id * 64 + 32 + col_id * 8]);
                }
                else
                {
                    A_reg[m_tile][0].int4_array[k_tile] =
                        *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
                }
            }
        }

        float a_scale_tail[WARP_M / MFMA_M][4][WARP_K / MFMA_K];
        if constexpr (GroupwisePost && AGroupScale)
        {
            const int group_k0 = k_start / 32;
            load_as_tile<WARP_M, WARP_K, MFMA_M, MFMA_K, BLOCK_M>(
                a_scale_tail,
                input_scale_ptr,
                sorted_token_ids_offset,
                group_k0,
                bidx,
                col_id,
                sorted_token_lens,
                top_k,
                max_n_len_offset,
                scale_A_stride_m,
                scale_A_stride_k);
        }

#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
        {
            const Element *cur_weight_ptr = weight_ptr + n_loop * BLOCK_N * 32;
            union
            {
                vec<int, 4> v;
                vec<int, 2> v2[2];
                uint32_t i[4];
            } weight_reg_tmp[WARP_N / 32];

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    if constexpr (GroupwisePost)
                    {
                        buffer_load_reg_dwordx2_w4a8(
                            cur_weight_ptr,
                            weight_reg_tmp[n_tile].v2[0],
                            k_tile * 16 * size_n + k_start_b,
                            g_row_B[n_tile]);
                        buffer_load_reg_dwordx2_w4a8(
                            cur_weight_ptr,
                            weight_reg_tmp[n_tile].v2[1],
                            16 * 32 + k_start_b,
                            g_row_B[n_tile]);
                    }
                    else
                    {
                        buffer_load_reg_dwordx4_w4a8(
                            cur_weight_ptr,
                            weight_reg_tmp[n_tile].v,
                            k_tile * 32 * size_n + k_start_b,
                            g_row_B[n_tile]);
                    }
                }
            }

            vmcnt_only_wait(0);
            __builtin_amdgcn_sched_barrier(0);

            float b_scale_tail[WARP_N / 32][2][WARP_K / MFMA_K];
            if constexpr (GroupwisePost)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
                        const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int group_k = (k_start + k_tile * MFMA_K) / 32;
                            b_scale_tail[n_tile][it][k_tile] = wfp4a8_e8m0_to_float(
                                weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                        }
                    }
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] =
                            *(uint32_t *)(&(weight_reg_tmp[n_tile].i[GroupwisePost ? (k_tile * 2 + it) : (it * 2 + k_tile)]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first;
                        uint32_t packed_val_second;
                        if (fp4x2_lut_lds != nullptr)
                        {
                            packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        }
                        else
                        {
                            packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(packed_val[k_tile]);
                            packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(packed_val[k_tile]);
                        }
                        B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                vec4_fp32 partial = {0, 0, 0, 0};
                                partial = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                    partial);
                                const float scale = b_scale_tail[n_tile][it][k_tile];
                                if constexpr (AGroupScale)
                                {
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        C_reg[n_loop][tile_idx][reg_id] +=
                                            partial[reg_id] * scale * a_scale_tail[m_tile][reg_id][k_tile];
                                    }
                                }
                                else
                                {
                                    C_reg[n_loop][tile_idx] += partial * scale;
                                }
                            }
                            else
                            {
                                C_reg[n_loop][tile_idx] =
                                    mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                        C_reg[n_loop][tile_idx]);
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    }
}

template <bool Is_store_A,
          int A_prefetch_level,
          int B_prefetch_level,
          int WARP_NUM,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGE,
          int GROUP_N,
          int GROUP_K,
          int SIZE_K,
          typename Element,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_prefill_wfp4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    const uint32_t *fp4x2_lut_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k,
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    float tmp[4][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 4;

    const int size_k = SIZE_K;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        int index = s_index + row_id * 2 * 8 + col_id * 32 * 8;

        g_row_B[n_tile] = index;
    }

    int A_index = warp_id;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(s_offset + col_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile] / top_k, max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < SIZE_K / 128; kloop++)
    {
        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            WFP4A8_BPIPE_BARRIER();
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }

            WFP4A8_BPIPE_BARRIER();

            i = 1;
            k_start_b += stage_offset_b;

            WFP4A8_BPIPE_BARRIER();

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }
            WFP4A8_BPIPE_BARRIER();

            vmcnt_only_wait(0);
            __syncthreads();
            WFP4A8_BPIPE_BARRIER();

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16) * 64 + +(i)*WARP_M / 16 * (16) * WARP_K;
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
                    }
                }
            }
            __syncthreads();
            WFP4A8_BPIPE_BARRIER();

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                i = 1;
                k_start_b += stage_offset_b;

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                WFP4A8_BPIPE_BARRIER();
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                WFP4A8_BPIPE_BARRIER();

                WFP4A8_BPIPE_BARRIER();

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                WFP4A8_BPIPE_BARRIER();

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

                WFP4A8_BPIPE_BARRIER();
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;
            }

            n_loop--;
            if (kloop < (SIZE_K / 128) - 1)
            {

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));

            if (kloop == (SIZE_K / 128) - 1)
            {
                vmcnt_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));

            if (kloop == (SIZE_K / 128) - 1)
            {
                vmcnt_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    wfp4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, g_row_A, g_row_B, n_loop_num);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                    }
                }
            }
        }
    }
}

template <bool Is_store_A,
          int A_prefetch_level,
          int B_prefetch_level,
          int WARP_NUM,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGE,
          int GROUP_N,
          int GROUP_K,
          int SIZE_K,
          typename Element,
          typename ElementAccum = int32_t,
          int N_LOOP_NUM = 4,
          bool GroupwisePost = false,
          bool FastScaleAddr = false,
          bool AGroupScale = true>
__forceinline__ __device__ void gemm_nt_marlin_prefill_2_wfp4a8_fixed_k64(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    const uint32_t *fp4x2_lut_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    const uint8_t *weight_scale_u8_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k,
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[N_LOOP_NUM][(WARP_N / 16)],
    float tmp[N_LOOP_NUM][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk,
    float *scale_lds = nullptr)
{

    constexpr int n_loop_num = N_LOOP_NUM;
    static_assert(SIZE_K % 64 == 0, "gemm_nt_marlin_prefill_2_wfp4a8_fixed_k64 requires K to be a multiple of 64");
    static_assert(WARP_K == 64, "gemm_nt_marlin_prefill_2_wfp4a8_fixed_k64 assumes WARP_K=64");
    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr bool UseScaleLds = false;
    constexpr int K_TILE_NUM = WARP_K / MFMA_K;
    constexpr int A_SCALE_LDS_FLOATS = BLOCK_M * K_TILE_NUM;
    constexpr int B_SCALE_LDS_BYTES = BLOCK_N * n_loop_num * K_TILE_NUM;
    constexpr int SCALE_SLOT_BYTES = A_SCALE_LDS_FLOATS * sizeof(float) + B_SCALE_LDS_BYTES;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    const int size_n = scale_B_stride_e;
    int A_index = warp_id;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];
    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(s_offset + col_id, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        constexpr int B_PACK_BYTES = GroupwisePost ? 4 : 8;
        constexpr int B_K_TILE_BYTES = 32;
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * B_K_TILE_BYTES;
        g_row_B[n_tile] = s_index + row_id * 2 * B_PACK_BYTES + col_id * 32 * B_PACK_BYTES;
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile], max_n_len_offset - 1)) * SIZE_K + row_id * 4;
    }

    constexpr int K_BLOCKS = SIZE_K / 64;
    {
        constexpr int k_block = 0;
        const int k_start = warp_k_id * WARP_K + k_block * 64;
        const int a_slot = 0;
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                inline_buffer_load_dword_lds(A_lds, g_input,
                                             (a_slot + m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                                             (k_tile * READ_K + k_start) / 4,
                                             g_row_A[m_tile] / 4);
            }
        }
        if constexpr (UseScaleLds)
        {
            load_scale_slot<
                BLOCK_M, BLOCK_N, WARP_M, WARP_N, WARP_K, MFMA_M, MFMA_K,
                n_loop_num, FastScaleAddr, SIZE_K>(
                reinterpret_cast<uint8_t *>(scale_lds),
                input_scale_ptr,
                weight_scale_u8_ptr,
                sorted_token_ids_offset,
                k_start / 32,
                bidx,
                warp_n_id,
                row_id,
                col_id,
                sorted_token_lens,
                top_k,
                max_n_len_offset,
                scale_A_stride_m,
                scale_A_stride_k,
                scale_B_stride_n,
                scale_B_stride_k);
        }
    }

#pragma unroll
    for (int k_block = 0; k_block < K_BLOCKS; k_block++)
    {
        const int k_start = warp_k_id * WARP_K + k_block * 64;
        const int k_start_b = warp_k_id * (WARP_K / 2) * size_n + k_block * 32 * size_n;
        const int a_slot = (k_block & 1) * WARP_M * WARP_K;
        uint8_t *scale_slot = reinterpret_cast<uint8_t *>(scale_lds) + (k_block & 1) * SCALE_SLOT_BYTES;

        vmcnt_only_wait(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        if (k_block + 1 < K_BLOCKS)
        {
            const int next_slot = ((k_block + 1) & 1) * WARP_M * WARP_K;
            const int next_k_start = warp_k_id * WARP_K + (k_block + 1) * 64;
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    inline_buffer_load_dword_lds(A_lds, g_input,
                                                 (next_slot + m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                                                 (k_tile * READ_K + next_k_start) / 4,
                                                 g_row_A[m_tile] / 4);
                }
            }
            if constexpr (UseScaleLds)
            {
                load_scale_slot<
                    BLOCK_M, BLOCK_N, WARP_M, WARP_N, WARP_K, MFMA_M, MFMA_K,
                    n_loop_num, FastScaleAddr, SIZE_K>(
                    reinterpret_cast<uint8_t *>(scale_lds) + ((k_block + 1) & 1) * SCALE_SLOT_BYTES,
                    input_scale_ptr,
                    weight_scale_u8_ptr,
                    sorted_token_ids_offset,
                    next_k_start / 32,
                    bidx,
                    warp_n_id,
                    row_id,
                    col_id,
                    sorted_token_lens,
                    top_k,
                    max_n_len_offset,
                    scale_A_stride_m,
                    scale_A_stride_k,
                    scale_B_stride_n,
                    scale_B_stride_k);
            }
        }

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                int s_index = m_tile * 16 * WARP_K;
                if constexpr (GroupwisePost)
                {
                    A_reg[m_tile][0].int8t_array[0] =
                        *(vec<Element, 8> *)(&A_lds[a_slot + s_index + row_id * 64 + col_id * 8]);
                    A_reg[m_tile][0].int8t_array[1] =
                        *(vec<Element, 8> *)(&A_lds[a_slot + s_index + row_id * 64 + 32 + col_id * 8]);
                }
                else
                {
                    A_reg[m_tile][0].int4_array[k_tile] =
                        *(vec<Element, 16> *)(&A_lds[a_slot + s_index + row_id * 64 + col_id * 16]);
                }
            }
        }

        float a_scale_reg_all[WARP_M / MFMA_M][4][WARP_K / MFMA_K];
        if constexpr (UseScaleLds)
        {
        }
        else if constexpr (GroupwisePost && AGroupScale)
        {
            const int group_k0 = k_start / 32;
            load_as_tile<WARP_M, WARP_K, MFMA_M, MFMA_K, BLOCK_M>(
                a_scale_reg_all,
                input_scale_ptr,
                sorted_token_ids_offset,
                group_k0,
                bidx,
                col_id,
                sorted_token_lens,
                top_k,
                max_n_len_offset,
                scale_A_stride_m,
                scale_A_stride_k);
        }

#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
        {
            const Element *cur_weight_ptr = weight_ptr + n_loop * BLOCK_N * 32;
            union
            {
                vec<int, 4> v;
                vec<int, 2> v2[2];
                uint32_t i[4];
            } weight_reg_tmp[WARP_N / 32];

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    if constexpr (GroupwisePost)
                    {
                        buffer_load_reg_dwordx2_w4a8(
                            cur_weight_ptr,
                            weight_reg_tmp[n_tile].v2[0],
                            k_tile * 16 * size_n + k_start_b,
                            g_row_B[n_tile]);
                        buffer_load_reg_dwordx2_w4a8(
                            cur_weight_ptr,
                            weight_reg_tmp[n_tile].v2[1],
                            16 * 32 + k_start_b,
                            g_row_B[n_tile]);
                    }
                    else
                    {
                        buffer_load_reg_dwordx4_w4a8(
                            cur_weight_ptr,
                            weight_reg_tmp[n_tile].v,
                            k_tile * 32 * size_n + k_start_b,
                            g_row_B[n_tile]);
                    }
                }
            }

            vmcnt_only_wait(0);
            __builtin_amdgcn_sched_barrier(0);

            float scale_reg_all[WARP_N / 32][2][WARP_K / MFMA_K];
            if constexpr (UseScaleLds)
            {
            }
            else if constexpr (GroupwisePost)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
                        const int logical_n =
                            BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int group_k = (k_start + k_tile * MFMA_K) / 32;
                            scale_reg_all[n_tile][it][k_tile] =
                                load_gs<FastScaleAddr, SIZE_K>(
                                    weight_scale_u8_ptr, logical_n, group_k, scale_B_stride_n, scale_B_stride_k);
                        }
                    }
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] =
                            *(uint32_t *)(&(weight_reg_tmp[n_tile].i[GroupwisePost ? (k_tile * 2 + it) : (it * 2 + k_tile)]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                vec4_fp32 partial = {0, 0, 0, 0};
                                partial = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                    partial);
                                float scale;
                                if constexpr (UseScaleLds)
                                {
                                    uint8_t *b_scale_u8_lds = scale_slot + A_SCALE_LDS_FLOATS * sizeof(float);
                                    const int local_n = warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    scale = wfp4a8_e8m0_to_float(
                                        b_scale_u8_lds[(n_loop * BLOCK_N + local_n) * K_TILE_NUM + k_tile]);
                                }
                                else
                                {
                                    scale = scale_reg_all[n_tile][it][k_tile];
                                }
                                if constexpr (AGroupScale)
                                {
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        float a_scale;
                                        if constexpr (UseScaleLds)
                                        {
                                            float *a_scale_lds = reinterpret_cast<float *>(scale_slot);
                                            const int m_local = m_tile * MFMA_M + reg_id * 4 + col_id;
                                            a_scale = a_scale_lds[m_local * K_TILE_NUM + k_tile];
                                        }
                                        else
                                        {
                                            a_scale = a_scale_reg_all[m_tile][reg_id][k_tile];
                                        }
                                        C_reg[n_loop][tile_idx][reg_id] +=
                                            partial[reg_id] * scale * a_scale;
                                    }
                                }
                                else
                                {
                                    C_reg[n_loop][tile_idx] += partial * scale;
                                }
                            }
                            else
                            {
                                C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                    C_reg[n_loop][tile_idx]);
                            }
                        }
                    }
                }
            }
        }
        vmcnt_only_wait(0);
        __syncthreads();
    }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        if constexpr (GroupwisePost)
                        {
                            tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                                C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id];
                        }
                        else
                        {
                            tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                                C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                        }
                    }
                }
            }
        }
    }
}

template <bool Is_store_A,
          int A_prefetch_level,
          int B_prefetch_level,
          int WARP_NUM,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGE,
          int GROUP_N,
          int GROUP_K,
          int SIZE_K,
          int N_LOOP_NUM,
          typename Element,
          typename ElementAccum = int32_t,
          bool GroupwisePost = false,
          bool GroupwisePre = false,
          bool KLoop64 = false,
          bool ShortScale = false,
          bool NoTmp = false,
          bool Ws12 = false,
          bool FastScaleAddr = false,
          bool SplitBLoadK32 = false,
          bool SplitARegK32 = false,
          bool KOuterScale = false,
          bool PairKScale = false,
          bool U8ScaleReg = false,
          bool FastBRow = false,
          bool AGroupScale = true>
__forceinline__ __device__ void gemm_nt_marlin_prefill_wfp4a8_gemm1n256(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    const uint32_t *fp4x2_lut_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k,
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[N_LOOP_NUM][(WARP_N / 16)],
    float tmp[N_LOOP_NUM][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk,
    const uint8_t *weight_scale_u8_ptr = nullptr,
    int logical_size_n = 0,
    float *scale_lds = nullptr

)
{

    constexpr int n_loop_num = N_LOOP_NUM;

    const int size_k = SIZE_K;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int K_TILE_NUM = WARP_K / MFMA_K;
    constexpr bool UseScaleLds = false;
    constexpr int A_SCALE_LDS_FLOATS = BLOCK_M * K_TILE_NUM;
    constexpr int B_SCALE_LDS_BYTES = BLOCK_N * n_loop_num * K_TILE_NUM;
    constexpr int SCALE_SLOT_BYTES = A_SCALE_LDS_FLOATS * sizeof(float) + B_SCALE_LDS_BYTES;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

    constexpr int B_PACK_BYTES = SplitBLoadK32 ? 4 : 8;
    constexpr int B_K_TILE_BYTES = 32;
    const int lane_b_base = row_id * 2 * B_PACK_BYTES + col_id * 32 * B_PACK_BYTES;
#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        const int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * B_K_TILE_BYTES;
        if constexpr (FastBRow)
        {
            g_row_B[n_tile] = lane_b_base + s_index;
        }
        else
        {
            g_row_B[n_tile] = s_index + row_id * 2 * B_PACK_BYTES + col_id * 32 * B_PACK_BYTES;
        }
    }

    const bool do_compute = !Ws12 || warp_id < 12;
    const bool do_load_a = !Ws12 || (warp_id >= 12 && warp_id < 16);
    int A_index = Ws12 ? (do_load_a ? warp_id - 12 : 0) : warp_id;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        sorted_token_ids_element[m_tile] = sorted_token_ids_offset[std::min(s_offset + col_id, int(sorted_token_lens - 1))];
        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile] / top_k, max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    if constexpr (KLoop64)
    {
        constexpr int B_N_LOOP_BYTES = BLOCK_N * 32;
        constexpr bool A2Stage = (STAGE == 2) && !Ws12;
        for (int k_block = 0; k_block < SIZE_K / 64; k_block++)
        {
            const int k_start = warp_k_id * WARP_K + k_block * BLOCK_K;
            const int k_start_b = warp_k_id * (WARP_K / 2) * size_n + k_block * (BLOCK_K / 2) * size_n;
            const int a_slot = (Ws12 || A2Stage) ? (k_block & 1) * WARP_M * WARP_K : 0;
            uint8_t *scale_slot = reinterpret_cast<uint8_t *>(scale_lds) +
                                  ((Ws12 || A2Stage) ? (k_block & 1) * SCALE_SLOT_BYTES : 0);

            if (do_load_a && (!(Ws12 || A2Stage) || k_block == 0))
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(
                            A_lds,
                            g_input,
                            (a_slot + m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                            (k_tile * READ_K + k_start) / 4,
                            g_row_A[m_tile] / 4);
                    }
                }
                if constexpr (UseScaleLds)
                {
                    load_scale_slot<
                        BLOCK_M, BLOCK_N, WARP_M, WARP_N, WARP_K, MFMA_M, MFMA_K,
                        n_loop_num, FastScaleAddr, SIZE_K>(
                        scale_slot,
                        input_scale_ptr,
                        weight_scale_u8_ptr,
                        sorted_token_ids_offset,
                        k_start / 32,
                        bidx,
                        warp_n_id,
                        row_id,
                        col_id,
                        sorted_token_lens,
                        top_k,
                        max_n_len_offset,
                        scale_A_stride_m,
                        scale_A_stride_k,
                        scale_B_stride_n,
                        scale_B_stride_k);
                }
            }

            vmcnt_only_wait(0);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);

            if constexpr (Ws12 || A2Stage)
            {
                if (do_load_a && k_block + 1 < SIZE_K / 64)
                {
                    const int next_slot = ((k_block + 1) & 1) * WARP_M * WARP_K;
                    const int next_k_start = warp_k_id * WARP_K + (k_block + 1) * BLOCK_K;
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            inline_buffer_load_dword_lds(
                                A_lds,
                                g_input,
                                (next_slot + m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                                (k_tile * READ_K + next_k_start) / 4,
                                g_row_A[m_tile] / 4);
                        }
                    }
                    if constexpr (UseScaleLds)
                    {
                        load_scale_slot<
                            BLOCK_M, BLOCK_N, WARP_M, WARP_N, WARP_K, MFMA_M, MFMA_K,
                            n_loop_num, FastScaleAddr, SIZE_K>(
                            reinterpret_cast<uint8_t *>(scale_lds) + ((k_block + 1) & 1) * SCALE_SLOT_BYTES,
                            input_scale_ptr,
                            weight_scale_u8_ptr,
                            sorted_token_ids_offset,
                            next_k_start / 32,
                            bidx,
                            warp_n_id,
                            row_id,
                            col_id,
                            sorted_token_lens,
                            top_k,
                            max_n_len_offset,
                            scale_A_stride_m,
                            scale_A_stride_k,
                            scale_B_stride_n,
                            scale_B_stride_k);
                    }
                }
            }

            if (do_compute)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        const int s_index = m_tile * 16 * WARP_K;
                        if constexpr (SplitARegK32)
                        {
                            A_reg[m_tile][0].int8t_array[0] =
                                *(vec<Element, 8> *)(&A_lds[a_slot + s_index + row_id * 64 + col_id * 8]);
                            A_reg[m_tile][0].int8t_array[1] =
                                *(vec<Element, 8> *)(&A_lds[a_slot + s_index + row_id * 64 + 32 + col_id * 8]);
                        }
                        else
                        {
                            A_reg[m_tile][0].int4_array[k_tile] =
                                *(vec<Element, 16> *)(&A_lds[a_slot + s_index + row_id * 64 + col_id * 16]);
                        }
                    }
                }

                float scale_reg_all[n_loop_num][WARP_N / 32][2][WARP_K / MFMA_K];
                uint8_t scale_u8_reg_all[n_loop_num][WARP_N / 32][2][WARP_K / MFMA_K];
                float a_scale_reg_all[WARP_M / MFMA_M][4][WARP_K / MFMA_K];
                if constexpr (UseScaleLds)
                {
                }
                else if constexpr (GroupwisePost && AGroupScale && (STAGE == 1 || STAGE == 2))
                {
                    const int group_k0 = (k_block * BLOCK_K + warp_k_id * WARP_K) / 32;
                    load_as_tile<WARP_M, WARP_K, MFMA_M, MFMA_K, BLOCK_M>(
                        a_scale_reg_all,
                        input_scale_ptr,
                        sorted_token_ids_offset,
                        group_k0,
                        bidx,
                        col_id,
                        sorted_token_lens,
                        top_k,
                        max_n_len_offset,
                        scale_A_stride_m,
                        scale_A_stride_k);
                }
                if constexpr (UseScaleLds)
                {
                }
                else if constexpr (GroupwisePost && (STAGE == 1 || (STAGE == 2 && !ShortScale)))
                {
                    const int group_k0 = (k_block * BLOCK_K + warp_k_id * WARP_K) / 32;
#pragma unroll
                    for (int scale_loop = 0; scale_loop < n_loop_num; scale_loop++)
                    {
                        const int logical_n_loop_base = BLOCK_N * scale_loop + warp_n_id * WARP_N + row_id * 2;
#pragma unroll
                        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                        {
                            const int logical_n_tile_base = logical_n_loop_base + n_tile * 32;
#pragma unroll
                            for (int it = 0; it < 2; it++)
                            {
                                const int logical_n = logical_n_tile_base + it;
                                const int scale_addr_base =
                                    FastScaleAddr
                                        ? logical_n * (SIZE_K / 32) + group_k0
                                        : logical_n * scale_B_stride_n + group_k0 * scale_B_stride_k;
#pragma unroll
                                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                {
                                    const int scale_addr = scale_addr_base + k_tile * (FastScaleAddr ? 1 : scale_B_stride_k);
                                    if constexpr (U8ScaleReg)
                                    {
                                        scale_u8_reg_all[scale_loop][n_tile][it][k_tile] =
                                            weight_scale_u8_ptr[scale_addr];
                                    }
                                    else
                                    {
                                        scale_reg_all[scale_loop][n_tile][it][k_tile] =
                                            load_gs_at(weight_scale_u8_ptr, scale_addr);
                                    }
                                }
                            }
                        }
                    }
                }

#pragma unroll
                for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
                {
                    const Element *cur_weight_ptr = weight_ptr + n_loop * B_N_LOOP_BYTES;

                    if constexpr (STAGE == 1)
                    {
                        union
                        {
                            vec<int, 4> v;
                            uint32_t i[4];
                        } weight_reg_tmp_k64[WARP_N / 32];

#pragma unroll
                        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                            {
                                buffer_load_reg_dwordx4_w4a8(
                                    cur_weight_ptr,
                                    weight_reg_tmp_k64[n_tile].v,
                                    k_tile * 32 * size_n + k_start_b,
                                    g_row_B[n_tile]);
                            }
                        }

                        vmcnt_only_wait(0);
                        __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                        {
#pragma unroll
                            for (int it = 0; it < 2; it++)
                            {
                                uint32_t packed_val[2];
#pragma unroll
                                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                {
                                    packed_val[k_tile] =
                                        *(uint32_t *)(&(weight_reg_tmp_k64[n_tile].i[it * 2 + k_tile]));
                                }
#pragma unroll
                                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                {
                                    uint32_t packed_val_first;
                                    uint32_t packed_val_second;
                                    if constexpr (GroupwisePre)
                                    {
                                        const int group_k =
                                            (k_block * BLOCK_K + warp_k_id * WARP_K + k_tile * MFMA_K) / 32;
                                        const int logical_n =
                                            BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                        const uint8_t scale =
                                            weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k];
                                        if (scale == 127)
                                        {
                                            packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                            packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                        }
                                        else
                                        {
                                            packed_val_first = wfp4a8_pack_high_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                            packed_val_second = wfp4a8_pack_low_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                        }
                                    }
                                    else
                                    {
                                        packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                        packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                    }
                                    B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][0] = *(int *)&packed_val_first;
                                    B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][1] = *(int *)&packed_val_second;
                                }
                            }
                        }

                        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                        {
#pragma unroll
                            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                            {
#pragma unroll
                                for (int it = 0; it < 2; it++)
                                {
#pragma unroll
                                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                    {
                                        const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                        if constexpr (GroupwisePost)
                                        {
                                            vec4_fp32 partial = {0, 0, 0, 0};
                                            partial = mmac_fp8<Element>(
                                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                                *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                                partial);
                                            const float scale = scale_reg_all[n_loop][n_tile][it][k_tile];
                                            if constexpr (AGroupScale)
                                            {
#pragma unroll
                                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                                {
                                                    C_reg[n_loop][tile_idx][reg_id] +=
                                                        partial[reg_id] * scale * a_scale_reg_all[m_tile][reg_id][k_tile];
                                                }
                                            }
                                            else
                                            {
                                                C_reg[n_loop][tile_idx] += partial * scale;
                                            }
                                        }
                                        else
                                        {
                                            C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                                *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                                C_reg[n_loop][tile_idx]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if constexpr (STAGE == 2)
                    {
                        if (n_loop == 0)
                        {
                            union
                            {
                                vec<int, 4> v;
                                vec<int, 2> v2[2];
                                uint32_t i[4];
                            } weight_reg_tmp_k64[WARP_N / 32][2];
                            constexpr int stage_i = 0;

#pragma unroll
                            for (int preload_loop = 0; preload_loop < 2; preload_loop++)
                            {
                                if constexpr (n_loop_num > 1)
                                {
                                    const int stage_b = preload_loop & 1;
                                    const Element *preload_weight_ptr = weight_ptr + preload_loop * B_N_LOOP_BYTES;
#pragma unroll
                                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                    {
#pragma unroll
                                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                                        {
                                            if constexpr (SplitBLoadK32)
                                            {
                                                buffer_load_reg_dwordx2_w4a8(
                                                    preload_weight_ptr,
                                                    weight_reg_tmp_k64[n_tile][stage_b].v2[0],
                                                    k_tile * 16 * size_n + k_start_b,
                                                    g_row_B[n_tile]);
                                                buffer_load_reg_dwordx2_w4a8(
                                                    preload_weight_ptr,
                                                    weight_reg_tmp_k64[n_tile][stage_b].v2[1],
                                                    16 * 32 + k_start_b,
                                                    g_row_B[n_tile]);
                                            }
                                            else
                                            {
                                                buffer_load_reg_dwordx4_w4a8(
                                                    preload_weight_ptr,
                                                    weight_reg_tmp_k64[n_tile][stage_b].v,
                                                    k_tile * 32 * size_n + k_start_b,
                                                    g_row_B[n_tile]);
                                            }
                                        }
                                    }
                                }
                            }

                            if constexpr (n_loop_num == 1)
                            {
#pragma unroll
                                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                {
#pragma unroll
                                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                                    {
                                        if constexpr (SplitBLoadK32)
                                        {
                                            buffer_load_reg_dwordx2_w4a8(
                                                weight_ptr,
                                                weight_reg_tmp_k64[n_tile][0].v2[0],
                                                k_tile * 16 * size_n + k_start_b,
                                                g_row_B[n_tile]);
                                            buffer_load_reg_dwordx2_w4a8(
                                                weight_ptr,
                                                weight_reg_tmp_k64[n_tile][0].v2[1],
                                                16 * 32 + k_start_b,
                                                g_row_B[n_tile]);
                                        }
                                        else
                                        {
                                            buffer_load_reg_dwordx4_w4a8(
                                                weight_ptr,
                                                weight_reg_tmp_k64[n_tile][0].v,
                                                k_tile * 32 * size_n + k_start_b,
                                                g_row_B[n_tile]);
                                        }
                                    }
                                }
                            }

#pragma unroll
                            for (int compute_loop = 0; compute_loop < n_loop_num; compute_loop++)
                            {
                                constexpr int keep_vmcnt = WARP_N / 32;
                                if constexpr (n_loop_num > 1)
                                {
                                    if (compute_loop + 1 < n_loop_num)
                                    {
                                        vmcnt_only_wait(keep_vmcnt);
                                    }
                                    else
                                    {
                                        vmcnt_only_wait(0);
                                    }
                                }
                                else
                                {
                                    vmcnt_only_wait(0);
                                }
                                __builtin_amdgcn_sched_barrier(0);

                                const int stage_b = compute_loop & 1;
#pragma unroll
                                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                {
#pragma unroll
                                    for (int it = 0; it < 2; it++)
                                    {
                                        uint32_t packed_val[2];
#pragma unroll
                                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                        {
                                            if constexpr (SplitBLoadK32)
                                            {
                                                packed_val[k_tile] =
                                                    *(uint32_t *)(&(weight_reg_tmp_k64[n_tile][stage_b].i[k_tile * 2 + it]));
                                            }
                                            else
                                            {
                                                packed_val[k_tile] =
                                                    *(uint32_t *)(&(weight_reg_tmp_k64[n_tile][stage_b].i[it * 2 + k_tile]));
                                            }
                                        }
#pragma unroll
                                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                        {
                                            uint32_t packed_val_first;
                                            uint32_t packed_val_second;
                                            if constexpr (GroupwisePre)
                                            {
                                                const int group_k =
                                                    (k_block * BLOCK_K + warp_k_id * WARP_K + k_tile * MFMA_K) / 32;
                                                const int logical_n =
                                                    BLOCK_N * compute_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                                const uint8_t scale =
                                                    weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k];
                                                if (scale == 127)
                                                {
                                                    packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                                    packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                                }
                                                else
                                                {
                                                    packed_val_first = wfp4a8_pack_high_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                                    packed_val_second = wfp4a8_pack_low_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                                }
                                            }
                                            else
                                            {
                                                packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                                packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                            }
                                            B_reg[n_tile * 2 + it][stage_b][stage_i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                                            B_reg[n_tile * 2 + it][stage_b][stage_i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                                        }
                                    }
                                }

                                const int next_loop = compute_loop + 2;
                                if (next_loop < n_loop_num)
                                {
                                    const int next_stage_b = next_loop & 1;
                                    const Element *next_weight_ptr = weight_ptr + next_loop * B_N_LOOP_BYTES;
#pragma unroll
                                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                    {
#pragma unroll
                                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                                        {
                                            if constexpr (SplitBLoadK32)
                                            {
                                                buffer_load_reg_dwordx2_w4a8(
                                                    next_weight_ptr,
                                                    weight_reg_tmp_k64[n_tile][next_stage_b].v2[0],
                                                    k_tile * 16 * size_n + k_start_b,
                                                    g_row_B[n_tile]);
                                                buffer_load_reg_dwordx2_w4a8(
                                                    next_weight_ptr,
                                                    weight_reg_tmp_k64[n_tile][next_stage_b].v2[1],
                                                    16 * 32 + k_start_b,
                                                    g_row_B[n_tile]);
                                            }
                                            else
                                            {
                                                buffer_load_reg_dwordx4_w4a8(
                                                    next_weight_ptr,
                                                    weight_reg_tmp_k64[n_tile][next_stage_b].v,
                                                    k_tile * 32 * size_n + k_start_b,
                                                    g_row_B[n_tile]);
                                            }
                                        }
                                    }
                                }

                                __builtin_amdgcn_sched_barrier(0);
                                if constexpr (GroupwisePost && ShortScale)
                                {
#pragma unroll
                                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                    {
#pragma unroll
                                        for (int it = 0; it < 2; it++)
                                        {
#pragma unroll
                                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                            {
                                                const int group_k =
                                                    (k_block * BLOCK_K + warp_k_id * WARP_K + k_tile * MFMA_K) / 32;
                                                const int logical_n =
                                                    BLOCK_N * compute_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                                const float scale =
                                                    load_gs<FastScaleAddr, SIZE_K>(
                                                        weight_scale_u8_ptr, logical_n, group_k, scale_B_stride_n, scale_B_stride_k);
#pragma unroll
                                                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                                                {
                                                    const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                                    vec4_fp32 partial = {0, 0, 0, 0};
                                                    partial = mmac_fp8<Element>(
                                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b][stage_i].int8t_array[k_tile]),
                                                        *(vec<Element, 8> *)(&A_reg[m_tile][stage_i].int8t_array[k_tile]),
                                                        partial);
                                                    if constexpr (AGroupScale)
                                                    {
#pragma unroll
                                                        for (int reg_id = 0; reg_id < 4; reg_id++)
                                                        {
                                                            const int a_row = load_arow(
                                                                sorted_token_ids_offset,
                                                                bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                                                sorted_token_lens,
                                                                top_k,
                                                                max_n_len_offset);
                                                            const float a_scale =
                                                                load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                                            C_reg[compute_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                                        }
                                                    }
                                                    else
                                                    {
                                                        C_reg[compute_loop][tile_idx] += partial * scale;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                else if constexpr (GroupwisePost && PairKScale)
                                {
#pragma unroll
                                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                                    {
#pragma unroll
                                        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                        {
#pragma unroll
                                            for (int it = 0; it < 2; it++)
                                            {
                                                const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                                vec4_fp32 partial0 = {0, 0, 0, 0};
                                                vec4_fp32 partial1 = {0, 0, 0, 0};
                                                partial0 = mmac_fp8<Element>(
                                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b][stage_i].int8t_array[0]),
                                                    *(vec<Element, 8> *)(&A_reg[m_tile][stage_i].int8t_array[0]),
                                                    partial0);
                                                partial1 = mmac_fp8<Element>(
                                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b][stage_i].int8t_array[1]),
                                                    *(vec<Element, 8> *)(&A_reg[m_tile][stage_i].int8t_array[1]),
                                                    partial1);
                                                if constexpr (AGroupScale)
                                                {
#pragma unroll
                                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                                    {
                                                        C_reg[compute_loop][tile_idx][reg_id] +=
                                                            partial0[reg_id] * scale_reg_all[compute_loop][n_tile][it][0] *
                                                                a_scale_reg_all[m_tile][reg_id][0] +
                                                            partial1[reg_id] * scale_reg_all[compute_loop][n_tile][it][1] *
                                                                a_scale_reg_all[m_tile][reg_id][1];
                                                    }
                                                }
                                                else
                                                {
                                                    C_reg[compute_loop][tile_idx] +=
                                                        partial0 * scale_reg_all[compute_loop][n_tile][it][0] +
                                                        partial1 * scale_reg_all[compute_loop][n_tile][it][1];
                                                }
                                            }
                                        }
                                    }
                                }
                                else if constexpr (GroupwisePost && KOuterScale)
                                {
#pragma unroll
                                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                    {
#pragma unroll
                                        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                        {
#pragma unroll
                                            for (int it = 0; it < 2; it++)
                                            {
                                                float scale;
                                                if constexpr (UseScaleLds)
                                                {
                                                    uint8_t *b_scale_u8_lds = scale_slot + A_SCALE_LDS_FLOATS * sizeof(float);
                                                    const int local_n =
                                                        warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                                    scale = wfp4a8_e8m0_to_float(
                                                        b_scale_u8_lds[(compute_loop * BLOCK_N + local_n) * K_TILE_NUM + k_tile]);
                                                }
                                                else if constexpr (U8ScaleReg)
                                                {
                                                    scale = wfp4a8_e8m0_to_float(
                                                        scale_u8_reg_all[compute_loop][n_tile][it][k_tile]);
                                                }
                                                else
                                                {
                                                    scale = scale_reg_all[compute_loop][n_tile][it][k_tile];
                                                }
#pragma unroll
                                                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                                                {
                                                    const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                                    vec4_fp32 partial = {0, 0, 0, 0};
                                                    partial = mmac_fp8<Element>(
                                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b][stage_i].int8t_array[k_tile]),
                                                        *(vec<Element, 8> *)(&A_reg[m_tile][stage_i].int8t_array[k_tile]),
                                                        partial);
                                                    if constexpr (AGroupScale)
                                                    {
#pragma unroll
                                                        for (int reg_id = 0; reg_id < 4; reg_id++)
                                                        {
                                                            float a_scale;
                                                            if constexpr (UseScaleLds)
                                                            {
                                                                float *a_scale_lds = reinterpret_cast<float *>(scale_slot);
                                                                const int m_local = m_tile * MFMA_M + reg_id * 4 + col_id;
                                                                a_scale = a_scale_lds[m_local * K_TILE_NUM + k_tile];
                                                            }
                                                            else
                                                            {
                                                                a_scale = a_scale_reg_all[m_tile][reg_id][k_tile];
                                                            }
                                                            C_reg[compute_loop][tile_idx][reg_id] +=
                                                                partial[reg_id] * scale * a_scale;
                                                        }
                                                    }
                                                    else
                                                    {
                                                        C_reg[compute_loop][tile_idx] += partial * scale;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                else
                                {
#pragma unroll
                                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                                    {
#pragma unroll
                                        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                                        {
#pragma unroll
                                            for (int it = 0; it < 2; it++)
                                            {
#pragma unroll
                                                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                                {
                                                    const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                                    if constexpr (GroupwisePost)
                                                    {
                                                        vec4_fp32 partial = {0, 0, 0, 0};
                                                        partial = mmac_fp8<Element>(
                                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b][stage_i].int8t_array[k_tile]),
                                                            *(vec<Element, 8> *)(&A_reg[m_tile][stage_i].int8t_array[k_tile]),
                                                            partial);
                                                        if constexpr (U8ScaleReg)
                                                        {
                                                            const float scale =
                                                                wfp4a8_e8m0_to_float(
                                                                    scale_u8_reg_all[compute_loop][n_tile][it][k_tile]);
                                                            if constexpr (AGroupScale)
                                                            {
#pragma unroll
                                                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                                                {
                                                                    C_reg[compute_loop][tile_idx][reg_id] +=
                                                                        partial[reg_id] * scale * a_scale_reg_all[m_tile][reg_id][k_tile];
                                                                }
                                                            }
                                                            else
                                                            {
                                                                C_reg[compute_loop][tile_idx] += partial * scale;
                                                            }
                                                        }
                                                        else
                                                        {
                                                            const float scale = scale_reg_all[compute_loop][n_tile][it][k_tile];
                                                            if constexpr (AGroupScale)
                                                            {
#pragma unroll
                                                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                                                {
                                                                    C_reg[compute_loop][tile_idx][reg_id] +=
                                                                        partial[reg_id] * scale * a_scale_reg_all[m_tile][reg_id][k_tile];
                                                                }
                                                            }
                                                            else
                                                            {
                                                                C_reg[compute_loop][tile_idx] += partial * scale;
                                                            }
                                                        }
                                                    }
                                                    else
                                                    {
                                                        C_reg[compute_loop][tile_idx] = mmac_fp8<Element>(
                                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b][stage_i].int8t_array[k_tile]),
                                                            *(vec<Element, 8> *)(&A_reg[m_tile][stage_i].int8t_array[k_tile]),
                                                            C_reg[compute_loop][tile_idx]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
            if constexpr (Ws12 || A2Stage)
            {
                vmcnt_only_wait(0);
            }
            __syncthreads();
        }

        if (do_compute)
        {
            if constexpr (!NoTmp)
            {
#pragma unroll
                for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
                {
#pragma unroll
                    for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
                    {
#pragma unroll
                        for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
                        {
#pragma unroll
                            for (int it = 0; it < 2; it++)
                            {
                                const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                {
                                    if constexpr (GroupwisePost || GroupwisePre)
                                    {
                                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                                            C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id];
                                    }
                                    else
                                    {
                                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                                            C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < SIZE_K / 128; kloop++)
    {
        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            vec<int, 2> v2[2];
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            WFP4A8_BPIPE_BARRIER();
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    if constexpr (GroupwisePost)
                    {
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                     k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                     16 * 32 + k_start_b, g_row_B[n_tile]);
                    }
                    else
                    {
                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
            }

            WFP4A8_BPIPE_BARRIER();

            i = 1;
            k_start_b += stage_offset_b;

            WFP4A8_BPIPE_BARRIER();

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    if constexpr (GroupwisePost)
                    {
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                     k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                     16 * 32 + k_start_b, g_row_B[n_tile]);
                    }
                    else
                    {
                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
            }
            WFP4A8_BPIPE_BARRIER();

            vmcnt_wait(2 * (WARP_N / 32));

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16) * 64 + +(i)*WARP_M / 16 * (16) * WARP_K;
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
                    }
                }
            }

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        if constexpr (GroupwisePost)
                        {
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                         k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                         16 * 32 + k_start_b, g_row_B[n_tile]);
                        }
                        else
                        {
                            buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                        }
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                i = 1;
                k_start_b += stage_offset_b;

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        if constexpr (GroupwisePost)
                        {
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                         k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                         16 * 32 + k_start_b, g_row_B[n_tile]);
                        }
                        else
                        {
                            buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                        }
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                WFP4A8_BPIPE_BARRIER();
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                WFP4A8_BPIPE_BARRIER();

                WFP4A8_BPIPE_BARRIER();

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first;
                            uint32_t packed_val_second;
                            if constexpr (GroupwisePre)
                            {
                                const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                const uint8_t scale = weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k];
                                if (scale == 127)
                                {
                                    packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                    packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                }
                                else
                                {
                                    packed_val_first = wfp4a8_pack_high_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                    packed_val_second = wfp4a8_pack_low_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                }
                            }
                            else
                            {
                                packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            }
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                WFP4A8_BPIPE_BARRIER();
                float scale_reg_i0[WARP_N / 32][2][WARP_K / MFMA_K];
                if constexpr (GroupwisePost && ShortScale)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                scale_reg_i0[n_tile][it][k_tile] =
                                    wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                            float scale = 1.0f;
                            if constexpr (GroupwisePost)
                            {
                                if constexpr (ShortScale)
                                {
                                    scale = scale_reg_i0[n_tile][it][k_tile];
                                }
                                else
                                {
                                    const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    scale = WFP4A8_SKIP_GROUP_SCALE ? 1.0f : load_gs<FastScaleAddr, SIZE_K>(weight_scale_u8_ptr, logical_n, group_k, scale_B_stride_n, scale_B_stride_k);
                                }
                            }
#pragma unroll
                            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                            {
                                const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                if constexpr (GroupwisePost)
                                {
                                    vec4_fp32 partial = {0, 0, 0, 0};
                                    if constexpr (ShortScale)
                                    {
                                        partial = mmac_fp8_lit<Element>(
                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                            partial);
                                    }
                                    else
                                    {
                                        partial = mmac_fp8<Element>(
                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                            partial);
                                    }
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        const int a_row = load_arow(
                                            sorted_token_ids_offset,
                                            bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                            sorted_token_lens,
                                            top_k,
                                            max_n_len_offset);
                                        const float a_scale =
                                            load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                        C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                    }
                                }
                                else
                                {
                                    C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        C_reg[n_loop][tile_idx]);
                                }
                            }
                        }
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                WFP4A8_BPIPE_BARRIER();

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first;
                            uint32_t packed_val_second;
                            if constexpr (GroupwisePre)
                            {
                                const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                const uint8_t scale = weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k];
                                if (scale == 127)
                                {
                                    packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                    packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                }
                                else
                                {
                                    packed_val_first = wfp4a8_pack_high_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                    packed_val_second = wfp4a8_pack_low_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                }
                            }
                            else
                            {
                                packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            }
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

                WFP4A8_BPIPE_BARRIER();
                float scale_reg_i1[WARP_N / 32][2][WARP_K / MFMA_K];
                if constexpr (GroupwisePost && ShortScale)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                scale_reg_i1[n_tile][it][k_tile] =
                                    wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                            float scale = 1.0f;
                            if constexpr (GroupwisePost)
                            {
                                if constexpr (ShortScale)
                                {
                                    scale = scale_reg_i1[n_tile][it][k_tile];
                                }
                                else
                                {
                                    const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    scale = WFP4A8_SKIP_GROUP_SCALE ? 1.0f : load_gs<FastScaleAddr, SIZE_K>(weight_scale_u8_ptr, logical_n, group_k, scale_B_stride_n, scale_B_stride_k);
                                }
                            }
#pragma unroll
                            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                            {
                                const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                if constexpr (GroupwisePost)
                                {
                                    vec4_fp32 partial = {0, 0, 0, 0};
                                    if constexpr (ShortScale)
                                    {
                                        partial = mmac_fp8_lit<Element>(
                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                            partial);
                                    }
                                    else
                                    {
                                        partial = mmac_fp8<Element>(
                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                            *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                            partial);
                                    }
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        const int a_row = load_arow(
                                            sorted_token_ids_offset,
                                            bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                            sorted_token_lens,
                                            top_k,
                                            max_n_len_offset);
                                        const float a_scale =
                                            load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                        C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                    }
                                }
                                else
                                {
                                    C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        C_reg[n_loop][tile_idx]);
                                }
                            }
                        }
                    }
                }

                WFP4A8_BPIPE_BARRIER();
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;
            }

            n_loop--;
            if (kloop < (SIZE_K / 128) - 1)
            {

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));

            if (kloop == (SIZE_K / 128) - 1)
            {
                vmcnt_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first;
                        uint32_t packed_val_second;
                        if constexpr (GroupwisePre)
                        {
                            const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                            const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                            const uint8_t scale = weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k];
                            if (scale == 127)
                            {
                                packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            }
                            else
                            {
                                packed_val_first = wfp4a8_pack_high_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                packed_val_second = wfp4a8_pack_low_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                            }
                        }
                        else
                        {
                            packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        }
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            float scale_reg_tail0[WARP_N / 32][2][WARP_K / MFMA_K];
            if constexpr (GroupwisePost && ShortScale)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                            const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                            scale_reg_tail0[n_tile][it][k_tile] =
                                wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                        }
                    }
                }
            }
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                        float scale = 1.0f;
                        if constexpr (GroupwisePost)
                        {
                            if constexpr (ShortScale)
                            {
                                scale = scale_reg_tail0[n_tile][it][k_tile];
                            }
                            else
                            {
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                scale = WFP4A8_SKIP_GROUP_SCALE ? 1.0f : load_gs<FastScaleAddr, SIZE_K>(weight_scale_u8_ptr, logical_n, group_k, scale_B_stride_n, scale_B_stride_k);
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                vec4_fp32 partial = {0, 0, 0, 0};
                                if constexpr (ShortScale)
                                {
                                    partial = mmac_fp8_lit<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                }
                                else
                                {
                                    partial = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                }
#pragma unroll
                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                {
                                    const int a_row = load_arow(
                                        sorted_token_ids_offset,
                                        bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                        sorted_token_lens,
                                        top_k,
                                        max_n_len_offset);
                                    const float a_scale =
                                        load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                    C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                }
                            }
                            else
                            {
                                C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][tile_idx]);
                            }
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));

            if (kloop == (SIZE_K / 128) - 1)
            {
                vmcnt_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first;
                        uint32_t packed_val_second;
                        if constexpr (GroupwisePre)
                        {
                            const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                            const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                            const uint8_t scale = weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k];
                            if (scale == 127)
                            {
                                packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            }
                            else
                            {
                                packed_val_first = wfp4a8_pack_high_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                packed_val_second = wfp4a8_pack_low_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                            }
                        }
                        else
                        {
                            packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        }
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            float scale_reg_tail1[WARP_N / 32][2][WARP_K / MFMA_K];
            if constexpr (GroupwisePost && ShortScale)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                            const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                            scale_reg_tail1[n_tile][it][k_tile] =
                                wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                        }
                    }
                }
            }
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                        float scale = 1.0f;
                        if constexpr (GroupwisePost)
                        {
                            if constexpr (ShortScale)
                            {
                                scale = scale_reg_tail1[n_tile][it][k_tile];
                            }
                            else
                            {
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                scale = WFP4A8_SKIP_GROUP_SCALE ? 1.0f : load_gs<FastScaleAddr, SIZE_K>(weight_scale_u8_ptr, logical_n, group_k, scale_B_stride_n, scale_B_stride_k);
                            }
                        }
#pragma unroll
                        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                vec4_fp32 partial = {0, 0, 0, 0};
                                if constexpr (ShortScale)
                                {
                                    partial = mmac_fp8_lit<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                }
                                else
                                {
                                    partial = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                }
#pragma unroll
                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                {
                                    const int a_row = load_arow(
                                        sorted_token_ids_offset,
                                        bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                        sorted_token_lens,
                                        top_k,
                                        max_n_len_offset);
                                    const float a_scale =
                                        load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                    C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                }
                            }
                            else
                            {
                                C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][tile_idx]);
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
                const int k_start_i = warp_k_id * WARP_K + kloop * 128 + i * stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(
                            A_lds,
                            g_input,
                            (m_tile * 16 * WARP_K + A_index * 4 * WARP_K) / 4,
                            (k_tile * READ_K + k_start_i) / 4,
                            g_row_A[m_tile] / 4);
                    }
                }

                vmcnt_only_wait(0);
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        const int s_index = m_tile * 16 * WARP_K;
                        A_reg[m_tile][0].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
                    }
                }

#pragma unroll
                for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
                {
                    const Element *cur_weight_ptr = weight_ptr + n_loop * BLOCK_N * 32;
                    union
                    {
                        vec<int, 4> v;
                        uint32_t i[4];
                    } weight_reg_tmp_s1[WARP_N / 32];
                    const int k_start_b_i = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n + i * stage_offset_b;

#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                        {
                            buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp_s1[n_tile].v, k_tile * 32 * size_n + k_start_b_i, g_row_B[n_tile]);
                        }
                    }

                    vmcnt_only_wait(0);
                    __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
                            uint32_t packed_val[2];
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp_s1[n_tile].i[it * 2 + k_tile]));
                            }
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                uint32_t packed_val_first;
                                uint32_t packed_val_second;
                                if constexpr (GroupwisePre)
                                {
                                    const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                    const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    const uint8_t scale = weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k];
                                    if (scale == 127)
                                    {
                                        packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                        packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                    }
                                    else
                                    {
                                        packed_val_first = wfp4a8_pack_high_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                        packed_val_second = wfp4a8_pack_low_fp4x4_e8m0_to_fp8(fp4x2_lut_lds, packed_val[k_tile], scale);
                                    }
                                }
                                else
                                {
                                    packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                    packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                                }
                                B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][0] = *(int *)&packed_val_first;
                                B_reg[n_tile * 2 + it][0][0].int2_array[k_tile][1] = *(int *)&packed_val_second;
                            }
                        }
                    }

                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                    {
#pragma unroll
                        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                        {
#pragma unroll
                            for (int it = 0; it < 2; it++)
                            {
#pragma unroll
                                for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                                {
                                    const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                    if constexpr (GroupwisePost)
                                    {
                                        vec4_fp32 partial = {0, 0, 0, 0};
                                        partial = mmac_fp8<Element>(
                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                            *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                            partial);
                                        const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                        const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                        const float scale = wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
#pragma unroll
                                        for (int reg_id = 0; reg_id < 4; reg_id++)
                                        {
                                            const int a_row = load_arow(
                                                sorted_token_ids_offset,
                                                bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                                sorted_token_lens,
                                                top_k,
                                                max_n_len_offset);
                                            const float a_scale =
                                                load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                            C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                        }
                                    }
                                    else
                                    {
                                        C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                            *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][0][0].int8t_array[k_tile]),
                                            *(vec<Element, 8> *)(&A_reg[m_tile][0].int8t_array[k_tile]),
                                            C_reg[n_loop][tile_idx]);
                                    }
                                }
                            }
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
                __syncthreads();
            }
        }
    }

    wfp4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, g_row_A, g_row_B, n_loop_num);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        if constexpr (GroupwisePost || GroupwisePre)
                        {
                            tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                                C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id];
                        }
                        else
                        {
                            tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                                C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                        }
                    }
                }
            }
        }
    }
}

template <bool Is_store_A,
          int A_prefetch_level,
          int B_prefetch_level,
          int WARP_NUM,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGE,
          int GROUP_N,
          int GROUP_K,
          int SIZE_K,
          typename Element,
          typename ElementAccum = int32_t,
          int N_LOOP_NUM = 4>
__forceinline__ __device__ void gemm_nt_marlin_prefill_2_wfp4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k,
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[N_LOOP_NUM][(WARP_N / 16)],
    float tmp[N_LOOP_NUM][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = N_LOOP_NUM;

    const int size_k = SIZE_K;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int A_index = warp_id;

    const int size_n = scale_B_stride_e;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(s_offset + col_id, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * 32;
        int index = s_index + row_id * 2 * 8 + col_id * 32 * 8;

        g_row_B[n_tile] = index;
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile], max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < SIZE_K / 128; kloop++)
    {

        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }

            __builtin_amdgcn_sched_barrier(0);

            i = 1;
            k_start_b += stage_offset_b;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);

            vmcnt_wait(2 * (WARP_N / 32));

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16) * 64 + +(i)*WARP_M / 16 * (16) * WARP_K;
                        A_reg[m_tile][i].int4_array[k_tile] = *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
                    }
                }
            }

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                i = 1;
                k_start_b += stage_offset_b;

                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                __builtin_amdgcn_sched_barrier(0);

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(packed_val[k_tile]);
                            uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(packed_val[k_tile]);
                            uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                            }
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;
            }

            n_loop--;

            if (kloop < (SIZE_K / 128) - 1)
            {
                __syncthreads();

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));
            }
            else
            {
                vmcnt_only_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(packed_val[k_tile]);
                        uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));
            }
            else
            {
                vmcnt_only_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[it * 2 + k_tile]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(packed_val[k_tile]);
                        uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it] = mmac_fp8<Element>(
                                *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                C_reg[n_loop][m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it]);
                        }
                    }
                }
            }

            WFP4A8_BPIPE_BARRIER();
            WFP4A8_BPIPE_BARRIER();
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    wfp4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, g_row_A, g_row_B, n_loop_num);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + it];
                    }
                }
            }
        }
    }
}

template <bool Is_store_A,
          int A_prefetch_level,
          int B_prefetch_level,
          int WARP_NUM,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGE,
          int GROUP_N,
          int GROUP_K,
          typename Element,
          bool GroupwisePost = false,
          bool AGroupScale = true,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_wfp4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    const uint32_t *fp4x2_lut_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    const uint8_t *weight_scale_u8_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][STAGE],
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k,
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx)
{

    int size_k = seqlen_A_stride;
    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    const int size_n = scale_B_stride_e;
    int k_start = warp_k_id * WARP_K;
    int k_start_b = warp_k_id * WARP_K / 2 * size_n;
    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * WARP_K / 2 * size_n;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        const int32_t sorted_token_ids_element = sorted_token_ids_offset[std::min(bidx * BLOCK_M + m_tile * MFMA_M + row_id, int(sorted_token_lens - 1))];
        const int token_id = sorted_token_ids_element / top_k;
        g_row_A[m_tile] = std::min(token_id, max_n_len_offset - 1) * size_k + col_id * 16;
    }

#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        constexpr int B_PACK_BYTES = 4;
        constexpr int B_K_TILE_BYTES = 32;
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * B_K_TILE_BYTES;
        int index = s_index + row_id * 2 * B_PACK_BYTES + col_id * 32 * B_PACK_BYTES;
        g_row_B[n_tile] = index;
    }

    vec<char, 8> vec8_ones = (vec<char, 8>){1, 1, 1, 1, 1, 1, 1, 1};
    vec<float, 4> vec4_ones = {0, 0, 0, 0};

    union
    {
        vec<int, 4> v;
        vec<int, 2> v2[2];
        uint32_t i[4];
    } weight_reg_tmp[WARP_N / 32][WARP_K / READ_K][STAGE];

#pragma unroll
    for (int i = 0; i < STAGE - 1; ++i)
    {
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                if constexpr (GroupwisePost)
                {
                    const int a_base = g_row_A[m_tile] - col_id * 16 + k_tile * 64 + k_start;
                    A_reg[m_tile][i].int2_array[k_tile * 2] =
                        *(vec<int, 2> *)&(input_ptr[a_base + col_id * 8]);
                    A_reg[m_tile][i].int2_array[k_tile * 2 + 1] =
                        *(vec<int, 2> *)&(input_ptr[a_base + 32 + col_id * 8]);
                }
                else
                {
                    A_reg[m_tile][i].int4_array[k_tile] =
                        *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + k_tile * 64 + k_start]);
                }
            }
        }

#pragma unroll
        for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
        {
#pragma unroll
            for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
            {
                weight_reg_tmp[n_tile][k_tile][i].v2[0] =
                    *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + k_start_b + g_row_B[n_tile]]);
                weight_reg_tmp[n_tile][k_tile][i].v2[1] =
                    *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + 16 * 32 + k_start_b + g_row_B[n_tile]]);
            }
        }
        k_start += stage_offset;
        k_start_b += stage_offset_b;
    }
    k_start -= stage_offset;
    k_start_b -= stage_offset_b;

    for (; k_start < size_k - stage_offset * STAGE; k_start += stage_offset * STAGE, k_start_b += stage_offset_b * STAGE)
    {

#pragma unroll
        for (int i = 0; i < STAGE; ++i)
        {
#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    if constexpr (GroupwisePost)
                    {
                        const int a_base =
                            g_row_A[m_tile] - col_id * 16 + (i + 1) * stage_offset + k_tile * 64 + k_start;
                        A_reg[m_tile][(i + STAGE - 1) % STAGE].int2_array[k_tile * 2] =
                            *(vec<int, 2> *)&(input_ptr[a_base + col_id * 8]);
                        A_reg[m_tile][(i + STAGE - 1) % STAGE].int2_array[k_tile * 2 + 1] =
                            *(vec<int, 2> *)&(input_ptr[a_base + 32 + col_id * 8]);
                    }
                    else
                    {
                        A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile] =
                            *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start]);
                    }
                }
            }

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {
                    weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v2[0] =
                        *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                    weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v2[1] =
                        *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + 16 * 32 + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                }
            }
            vmcnt_wait(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                    for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                    {
#pragma unroll
                        for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                        {
                            const int packed_idx = k_tile_second * 2 + it;
                            packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][i].i[packed_idx]));
                        }
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {

                        uint32_t packed_val_first;
                        uint32_t packed_val_second;
                        packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                vec4_fp32 partial = {0, 0, 0, 0};
                                partial = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    partial);
                                const int group_k =
                                    (k_start - (STAGE - 2) * stage_offset + i * stage_offset + k_tile * MFMA_K) / 32;
                                const int logical_n =
                                    warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                const float scale = wfp4a8_e8m0_to_float(
                                    weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                                if constexpr (AGroupScale)
                                {
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        const int a_row = load_arow(
                                            sorted_token_ids_offset,
                                            bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                            sorted_token_lens,
                                            top_k,
                                            max_n_len_offset);
                                        const float a_scale =
                                            load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                        C_reg[0][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                    }
                                }
                                else
                                {
                                    C_reg[0][tile_idx] += partial * scale;
                                }
                            }
                            else
                            {
                                C_reg[0][tile_idx] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                    C_reg[0][tile_idx]);
                            }
                        }
                    }
                }
            }
        }
    }

    if (k_start < size_k)
    {
        int i = 0;
        if constexpr (STAGE == 4)
        {
            int epilogue_tile = (size_k / BLOCK_K - (STAGE - 1)) % STAGE;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        if constexpr (GroupwisePost)
                        {
                            const int a_base =
                                g_row_A[m_tile] - col_id * 16 + (i + 1) * stage_offset + k_tile * 64 + k_start;
                            A_reg[m_tile][(i + STAGE - 1) % STAGE].int2_array[k_tile * 2] =
                                *(vec<int, 2> *)&(input_ptr[a_base + col_id * 8]);
                            A_reg[m_tile][(i + STAGE - 1) % STAGE].int2_array[k_tile * 2 + 1] =
                                *(vec<int, 2> *)&(input_ptr[a_base + 32 + col_id * 8]);
                        }
                        else
                        {
                            A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile] =
                                *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start]);
                        }
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v2[0] =
                            *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                        weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v2[1] =
                            *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + 16 * 32 + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                    }
                }
                vmcnt_wait(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                        for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                        {
#pragma unroll
                            for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                            {
                                const int packed_idx = k_tile_second * 2 + it;
                                packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][i].i[packed_idx]));
                            }
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {

                            uint32_t packed_val_first;
                            uint32_t packed_val_second;
                            packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                if constexpr (GroupwisePost)
                                {
                                    vec4_fp32 partial = {0, 0, 0, 0};
                                    partial = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                    const int group_k =
                                        (k_start - (STAGE - 2) * stage_offset + i * stage_offset + k_tile * MFMA_K) / 32;
                                    const int logical_n =
                                        warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    const float scale = wfp4a8_e8m0_to_float(
                                        weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                                    if constexpr (AGroupScale)
                                    {
#pragma unroll
                                        for (int reg_id = 0; reg_id < 4; reg_id++)
                                        {
                                            const int a_row = load_arow(
                                                sorted_token_ids_offset,
                                                bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                                sorted_token_lens,
                                                top_k,
                                                max_n_len_offset);
                                            const float a_scale =
                                                load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                            C_reg[0][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                        }
                                    }
                                    else
                                    {
                                        C_reg[0][tile_idx] += partial * scale;
                                    }
                                }
                                else
                                {
                                    C_reg[0][tile_idx] = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                        C_reg[0][tile_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            constexpr int epilogue_tile = 1;

#pragma unroll
            for (; i < epilogue_tile; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        if constexpr (GroupwisePost)
                        {
                            const int a_base =
                                g_row_A[m_tile] - col_id * 16 + (i + 1) * stage_offset + k_tile * 64 + k_start;
                            A_reg[m_tile][(i + STAGE - 1) % STAGE].int2_array[k_tile * 2] =
                                *(vec<int, 2> *)&(input_ptr[a_base + col_id * 8]);
                            A_reg[m_tile][(i + STAGE - 1) % STAGE].int2_array[k_tile * 2 + 1] =
                                *(vec<int, 2> *)&(input_ptr[a_base + 32 + col_id * 8]);
                        }
                        else
                        {
                            A_reg[m_tile][(i + STAGE - 1) % STAGE].int4_array[k_tile] =
                                *(vec<int, 4> *)&(input_ptr[g_row_A[m_tile] + (i + 1) * stage_offset + k_tile * 64 + k_start]);
                        }
                    }
                }

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v2[0] =
                            *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                        weight_reg_tmp[n_tile][k_tile][(i + STAGE - 1) % STAGE].v2[1] =
                            *(vec<int, 2> *)(&weight_ptr[k_tile * 32 * size_n + 16 * 32 + (i + 1) * stage_offset_b + k_start_b + g_row_B[n_tile]]);
                    }
                }
                vmcnt_wait(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                        for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                        {
#pragma unroll
                            for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                            {
                                const int packed_idx = k_tile_second * 2 + it;
                                packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][i].i[packed_idx]));
                            }
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {

                            uint32_t packed_val_first;
                            uint32_t packed_val_second;
                            packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {
                                const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                if constexpr (GroupwisePost)
                                {
                                    vec4_fp32 partial = {0, 0, 0, 0};
                                    partial = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                    const int group_k =
                                        (k_start - (STAGE - 2) * stage_offset + i * stage_offset + k_tile * MFMA_K) / 32;
                                    const int logical_n =
                                        warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    const float scale = wfp4a8_e8m0_to_float(
                                        weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                                    if constexpr (AGroupScale)
                                    {
#pragma unroll
                                        for (int reg_id = 0; reg_id < 4; reg_id++)
                                        {
                                            const int a_row = load_arow(
                                                sorted_token_ids_offset,
                                                bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                                sorted_token_lens,
                                                top_k,
                                                max_n_len_offset);
                                            const float a_scale =
                                                load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                            C_reg[0][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                        }
                                    }
                                    else
                                    {
                                        C_reg[0][tile_idx] += partial * scale;
                                    }
                                }
                                else
                                {
                                    C_reg[0][tile_idx] = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][i].int8t_array[k_tile]),
                                        C_reg[0][tile_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int ii = 1; ii < STAGE; ++ii)
        {
            vmcnt_wait(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[WARP_K / MFMA_K];
#pragma unroll
                    for (int k_tile_first = 0; k_tile_first < WARP_K / READ_K; k_tile_first++)
                    {
#pragma unroll
                        for (int k_tile_second = 0; k_tile_second < READ_K / MFMA_K; k_tile_second++)
                        {
                            const int packed_idx = k_tile_second * 2 + it;
                            packed_val[k_tile_first * (READ_K / MFMA_K) + k_tile_second] = *(uint32_t *)(&(weight_reg_tmp[n_tile][k_tile_first][(i + ii - 1) % STAGE].i[packed_idx]));
                        }
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {

                        uint32_t packed_val_first;
                        uint32_t packed_val_second;
                        packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        const int stage_idx = (i + ii - 1) % STAGE;
                        B_reg[n_tile * 2 + it][stage_idx].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_idx].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                const int stage_idx = (i + ii - 1) % STAGE;
                                vec4_fp32 partial = {0, 0, 0, 0};
                                partial = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_idx].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][stage_idx].int8t_array[k_tile]),
                                    partial);
                                const int group_k =
                                    (k_start - (STAGE - 2) * stage_offset +
                                     (i + ii - 1) * stage_offset + k_tile * MFMA_K) /
                                    32;
                                const int logical_n =
                                    warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                const float scale = wfp4a8_e8m0_to_float(
                                    weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
                                if constexpr (AGroupScale)
                                {
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        const int a_row = load_arow(
                                            sorted_token_ids_offset,
                                            bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                            sorted_token_lens,
                                            top_k,
                                            max_n_len_offset);
                                        const float a_scale =
                                            load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                        C_reg[0][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                    }
                                }
                                else
                                {
                                    C_reg[0][tile_idx] += partial * scale;
                                }
                            }
                            else
                            {
                                C_reg[0][tile_idx] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&A_reg[m_tile][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][(i + ii - 1) % STAGE].int8t_array[k_tile]),
                                    C_reg[0][tile_idx]);
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
    }

    extern __shared__ float out_smem[];
    if constexpr (warp_k_num > 1)
    {

        constexpr int pading_n = WARP_N + 1;

        if constexpr (GroupwisePost)
        {
            if (warp_k_id > 0)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            const int n_idx = n_tile * 32 + row_id * 2 + it;
#pragma unroll
                            for (int reg_id = 0; reg_id < 4; reg_id++)
                            {
                                const int m_idx = m_tile * MFMA_M + reg_id * 4 + col_id;
                                out_smem[m_idx * pading_n + n_idx +
                                         (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * WARP_M] =
                                    C_reg[0][tile_idx][reg_id];
                            }
                        }
                    }
                }
            }

            __syncthreads();

            if (warp_k_id == 0)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            const int n_idx = n_tile * 32 + row_id * 2 + it;
#pragma unroll
                            for (int k_tile = 0; k_tile < warp_k_num - 1; k_tile++)
                            {
#pragma unroll
                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                {
                                    const int m_idx = m_tile * MFMA_M + reg_id * 4 + col_id;
                                    C_reg[0][tile_idx][reg_id] +=
                                        out_smem[m_idx * pading_n + n_idx +
                                                 (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * WARP_M];
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if (warp_k_id > 0)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {

                        *(vec4_fp32 *)(&out_smem[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (warp_k_id - 1 + warp_n_id * (warp_k_num - 1)) * pading_n * 16]) = C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile];
                    }
                }
            }

            __syncthreads();

            if (warp_k_id == 0)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / MFMA_N; n_tile++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < warp_k_num - 1; k_tile++)
                        {
                            vec4_fp32 temp = *(vec4_fp32 *)(&out_smem[(m_tile * MFMA_M + row_id) * pading_n + n_tile * 16 + col_id * 4 + (k_tile + warp_n_id * (warp_k_num - 1)) * pading_n * 16]);
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                            {
                                C_reg[0][m_tile * (WARP_N / MFMA_N) + n_tile][i] += temp[i];
                            }
                        }
                    }
                }
            }
        }
    }
}

template <bool Is_store_A,
          int A_prefetch_level,
          int B_prefetch_level,
          int WARP_NUM,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int WARP_M,
          int WARP_N,
          int WARP_K,
          int STAGE,
          int GROUP_N,
          int GROUP_K,
          int SIZE_K,
          typename Element,
          bool GroupwisePost = false,
          bool AGroupScale = true,
          typename ElementAccum = int32_t>
__forceinline__ __device__ void gemm_nt_marlin_decode_2_wfp4a8(
    const Element *input_ptr,
    const Element *weight_ptr,
    Element *A_lds,
    Element *B_lds,
    const uint32_t *fp4x2_lut_lds,
    float *input_scale_ptr,
    float *weight_scale_ptr,
    const uint8_t *weight_scale_u8_ptr,
    int max_n_len_offset,
    union_vec_opt<Element, WARP_K / 4> A_reg[][STAGE],
    union_vec_opt<Element, WARP_K / 4> B_reg[][2][STAGE],
    vec4_fp32 C_reg[][(WARP_M / 16) * (WARP_N / 16)],
    int warp_id,
    int seqlen_A_stride,
    int seqlen_B_stride,
    int scale_A_stride_m,
    int scale_A_stride_k,
    int scale_B_stride_e,
    int scale_B_stride_n,
    int scale_B_stride_k,
    int top_k,
    const int32_t *sorted_token_ids_offset,
    int sorted_token_lens,
    const int32_t expert_id,
    const int bidx,
    float weight_dot_a_scale[WARP_M / 16][4],
    float b_scale[4][(WARP_N / 16)],
    float tmp[4][WARP_M / 16][4][WARP_N / 16],
    uint32_t real_topk

)
{

    constexpr int n_loop_num = 4;

    const int size_k = SIZE_K;

    int lane_id = threadIdx.x & 63;
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    constexpr int MFMA_M = 16;
    constexpr int MFMA_N = 16;
    constexpr int MFMA_K = 32;
    constexpr int READ_K = 64;
    constexpr int warp_k_num = BLOCK_K / WARP_K;
    constexpr int warp_n_num = BLOCK_N / WARP_N;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;

    int A_index = warp_id;

    const int size_n = seqlen_B_stride;

    const int stage_offset = warp_k_num * WARP_K;
    const int stage_offset_b = warp_k_num * (WARP_K / 2) * size_n;
    const int lds_stage_offset = WARP_M * WARP_K;

    int g_row_A[WARP_M / MFMA_M];
    int g_row_B[WARP_N / MFMA_N];

    auto g_input = tcp_cache_swizzle_func_b8<128, Element>(input_ptr);
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids_offset);
    int32_t sorted_token_ids_element[WARP_M / MFMA_M];

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
        int s_offset = bidx * BLOCK_M + m_tile * MFMA_M + A_index * 4;
        inline_buffer_load_dword(sorted_token_ids_element[m_tile], std::min(s_offset + col_id, int(sorted_token_lens - 1)), g_sorted_token_ids_offset, 0);
    }

    constexpr int B_PACK_BYTES = GroupwisePost ? 4 : 8;
    constexpr int B_K_TILE_BYTES = 32;
    const int lane_b_base = row_id * 2 * B_PACK_BYTES + col_id * 32 * B_PACK_BYTES;
#pragma unroll
    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
    {
        int s_index = (warp_n_id * (WARP_N / 32) + n_tile) * 32 * B_K_TILE_BYTES;
        g_row_B[n_tile] = s_index + lane_b_base;
    }

    vmcnt_only_wait(0);

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {

        g_row_A[m_tile] = (std::min(sorted_token_ids_element[m_tile], max_n_len_offset - 1)) * size_k + row_id * 4;
    }

    int kloop = 0;
    int k_start = warp_k_id * (WARP_K) + kloop * (128);

    int i = 0;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    i = 1;

    k_start += stage_offset;

#pragma unroll
    for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
    {
#pragma unroll
        for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
        {
            inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
        }
    }

    for (kloop; kloop < SIZE_K / 128; kloop++)
    {

        int k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (128 / 2) * size_n;

        union
        {
            vec<int, 4> v;
            vec<int, 2> v2[2];
            uint32_t i[4];
        } weight_reg_tmp[WARP_N / 32][2][STAGE];

        if constexpr (STAGE == 2)
        {

            const Element *cur_weight_ptr = weight_ptr;

            int i = 0;
            int stage_b_flag = 0;
            int n_loop = 0;

            WFP4A8_BPIPE_BARRIER();
#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    if constexpr (GroupwisePost)
                    {
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                     k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                     16 * 32 + k_start_b, g_row_B[n_tile]);
                    }
                    else
                    {
                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
            }

            WFP4A8_BPIPE_BARRIER();

            i = 1;
            k_start_b += stage_offset_b;

            WFP4A8_BPIPE_BARRIER();

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                {

                    if constexpr (GroupwisePost)
                    {
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                     k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                        buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                     16 * 32 + k_start_b, g_row_B[n_tile]);
                    }
                    else
                    {
                        buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                    }
                }
            }
            WFP4A8_BPIPE_BARRIER();

            vmcnt_wait(2 * (WARP_N / 32));

#pragma unroll
            for (int i = 0; i < 2; i++)
            {
#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        int s_index = m_tile * (16) * 64 + +(i)*WARP_M / 16 * (16) * WARP_K;
                        if constexpr (GroupwisePost)
                        {
                            A_reg[m_tile][i].int8t_array[0] =
                                *(vec<Element, 8> *)(&A_lds[s_index + row_id * 64 + col_id * 8]);
                            A_reg[m_tile][i].int8t_array[1] =
                                *(vec<Element, 8> *)(&A_lds[s_index + row_id * 64 + 32 + col_id * 8]);
                        }
                        else
                        {
                            A_reg[m_tile][i].int4_array[k_tile] =
                                *(vec<Element, 16> *)(&A_lds[s_index + row_id * 64 + col_id * 16]);
                        }
                    }
                }
            }

            k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;

            n_loop++;

            for (n_loop; n_loop < n_loop_num; n_loop++)
            {

                stage_b_flag ^= 1;
                cur_weight_ptr += BLOCK_N * 32;
                int i = 0;

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        if constexpr (GroupwisePost)
                        {
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                         k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                         16 * 32 + k_start_b, g_row_B[n_tile]);
                        }
                        else
                        {
                            buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                        }
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                i = 1;
                k_start_b += stage_offset_b;

                WFP4A8_BPIPE_BARRIER();
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {

                        if constexpr (GroupwisePost)
                        {
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[0],
                                                         k_tile * 16 * size_n + k_start_b, g_row_B[n_tile]);
                            buffer_load_reg_dwordx2_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v2[1],
                                                         16 * 32 + k_start_b, g_row_B[n_tile]);
                        }
                        else
                        {
                            buffer_load_reg_dwordx4_w4a8(cur_weight_ptr, weight_reg_tmp[n_tile][stage_b_flag][i].v, k_tile * 32 * size_n + k_start_b, g_row_B[n_tile]);
                        }
                    }
                }
                WFP4A8_BPIPE_BARRIER();

                WFP4A8_BPIPE_BARRIER();
                vmcnt_only_wait(3 * (WARP_N / 32));

                n_loop--;
                i = 0;
                stage_b_flag ^= 1;

                WFP4A8_BPIPE_BARRIER();

                WFP4A8_BPIPE_BARRIER();

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[GroupwisePost ? (k_tile * 2 + it) : (it * 2 + k_tile)]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                if constexpr (GroupwisePost)
                                {
                                    vec4_fp32 partial = {0, 0, 0, 0};
                                    partial = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                    const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                    const float scale = wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        const int a_row = load_arow(
                                            sorted_token_ids_offset,
                                            bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                            sorted_token_lens,
                                            top_k,
                                            max_n_len_offset);
                                        if constexpr (AGroupScale)
                                        {
                                            const float a_scale =
                                                load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                            C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                        }
                                        else
                                        {
                                            C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale;
                                        }
                                    }
                                }
                                else
                                {
                                    C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        C_reg[n_loop][tile_idx]);
                                }
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                vmcnt_only_wait(2 * (WARP_N / 32));

                i = 1;

                __builtin_amdgcn_sched_barrier(0);

#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {

                        uint32_t packed_val[2];
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[GroupwisePost ? (k_tile * 2 + it) : (it * 2 + k_tile)]));
                        }
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                            B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                        }
                    }
                }

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                    {
#pragma unroll
                        for (int it = 0; it < 2; it++)
                        {
#pragma unroll
                            for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                            {

                                const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                                if constexpr (GroupwisePost)
                                {
                                    vec4_fp32 partial = {0, 0, 0, 0};
                                    partial = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        partial);
                                    const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                    const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                    const float scale = wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
#pragma unroll
                                    for (int reg_id = 0; reg_id < 4; reg_id++)
                                    {
                                        const int a_row = load_arow(
                                            sorted_token_ids_offset,
                                            bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                            sorted_token_lens,
                                            top_k,
                                            max_n_len_offset);
                                        if constexpr (AGroupScale)
                                        {
                                            const float a_scale =
                                                load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                            C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                        }
                                        else
                                        {
                                            C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale;
                                        }
                                    }
                                }
                                else
                                {
                                    C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                        *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                        *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                        C_reg[n_loop][tile_idx]);
                                }
                            }
                        }
                    }
                }

                __builtin_amdgcn_sched_barrier(0);
                stage_b_flag ^= 1;
                n_loop++;

                k_start_b = warp_k_id * (WARP_K / 2) * size_n + kloop * (WARP_K * 2 / 2) * size_n;
            }

            n_loop--;

            if (kloop < (SIZE_K / 128) - 1)
            {

                kloop++;
                k_start = warp_k_id * (WARP_K) + kloop * (128);

                i = 0;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (0) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }

                i = 1;

                k_start += stage_offset;

#pragma unroll
                for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
                {
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / READ_K; k_tile++)
                    {
                        inline_buffer_load_dword_lds(A_lds, g_input, (m_tile * (16) * WARP_K + A_index * 4 * WARP_K) / 4 + (1) * (WARP_M / 16) * (16) * WARP_K / 4, (k_tile * READ_K + k_start) / 4, (g_row_A[m_tile]) / 4);
                    }
                }
                kloop--;
            }

            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait((WARP_N / 32) + (2 * (WARP_M / MFMA_M) * (WARP_K / READ_K)));
            }
            else
            {
                vmcnt_only_wait((WARP_N / 32));
            }

            i = 0;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[GroupwisePost ? (k_tile * 2 + it) : (it * 2 + k_tile)]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                vec4_fp32 partial = {0, 0, 0, 0};
                                partial = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    partial);
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                const float scale = wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
#pragma unroll
                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                {
                                    const int a_row = load_arow(
                                        sorted_token_ids_offset,
                                        bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                        sorted_token_lens,
                                        top_k,
                                        max_n_len_offset);
                                    if constexpr (AGroupScale)
                                    {
                                        const float a_scale =
                                            load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                        C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                    }
                                    else
                                    {
                                        C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale;
                                    }
                                }
                            }
                            else
                            {
                                C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][tile_idx]);
                            }
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            if (kloop < (SIZE_K / 128) - 1)
            {
                vmcnt_only_wait(2 * (WARP_M / MFMA_M) * (WARP_K / READ_K));
            }
            else
            {
                vmcnt_only_wait(0);
            }

            i = 1;

            __builtin_amdgcn_sched_barrier(0);

#pragma unroll
            for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    uint32_t packed_val[2];
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        packed_val[k_tile] = *(uint32_t *)(&(weight_reg_tmp[n_tile][stage_b_flag][i].i[GroupwisePost ? (k_tile * 2 + it) : (it * 2 + k_tile)]));
                    }
#pragma unroll
                    for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                    {
                        uint32_t packed_val_first = wfp4a8_pack_high_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        uint32_t packed_val_second = wfp4a8_pack_low_fp4x4_to_fp8(fp4x2_lut_lds, packed_val[k_tile]);
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][0] = *(int *)&packed_val_first;
                        B_reg[n_tile * 2 + it][stage_b_flag][i].int2_array[k_tile][1] = *(int *)&packed_val_second;
                    }
                }
            }

#pragma unroll
            for (int m_tile = 0; m_tile < WARP_M / MFMA_M; m_tile++)
            {
#pragma unroll
                for (int n_tile = 0; n_tile < WARP_N / 32; n_tile++)
                {
#pragma unroll
                    for (int it = 0; it < 2; it++)
                    {
#pragma unroll
                        for (int k_tile = 0; k_tile < WARP_K / MFMA_K; k_tile++)
                        {
                            const int tile_idx = m_tile * (WARP_N / MFMA_N) + n_tile * 2 + it;
                            if constexpr (GroupwisePost)
                            {
                                vec4_fp32 partial = {0, 0, 0, 0};
                                partial = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    partial);
                                const int logical_n = BLOCK_N * n_loop + warp_n_id * WARP_N + n_tile * 32 + row_id * 2 + it;
                                const int group_k = (kloop * 128 + warp_k_id * WARP_K + i * stage_offset + k_tile * MFMA_K) / 32;
                                const float scale = wfp4a8_e8m0_to_float(weight_scale_u8_ptr[logical_n * scale_B_stride_n + group_k * scale_B_stride_k]);
#pragma unroll
                                for (int reg_id = 0; reg_id < 4; reg_id++)
                                {
                                    const int a_row = load_arow(
                                        sorted_token_ids_offset,
                                        bidx * BLOCK_M + m_tile * MFMA_M + reg_id * 4 + col_id,
                                        sorted_token_lens,
                                        top_k,
                                        max_n_len_offset);
                                    if constexpr (AGroupScale)
                                    {
                                        const float a_scale =
                                            load_as(input_scale_ptr, a_row, group_k, scale_A_stride_m, scale_A_stride_k);
                                        C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale * a_scale;
                                    }
                                    else
                                    {
                                        C_reg[n_loop][tile_idx][reg_id] += partial[reg_id] * scale;
                                    }
                                }
                            }
                            else
                            {
                                C_reg[n_loop][tile_idx] = mmac_fp8<Element>(
                                    *(vec<Element, 8> *)(&B_reg[n_tile * 2 + it][stage_b_flag][i].int8t_array[k_tile]),
                                    *(vec<Element, 8> *)(&A_reg[m_tile][i].int8t_array[k_tile]),
                                    C_reg[n_loop][tile_idx]);
                            }
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr (STAGE == 1)
        {
            ;
        }
    }

    wfp4a8_mmac_tail64<WARP_NUM, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, STAGE, SIZE_K, Element, GroupwisePost, AGroupScale>(
        input_ptr, weight_ptr, A_lds, A_reg, B_reg, C_reg, warp_id, size_n, g_row_A, g_row_B, n_loop_num,
        weight_scale_u8_ptr, scale_B_stride_n, scale_B_stride_k, fp4x2_lut_lds,
        input_scale_ptr, scale_A_stride_m, scale_A_stride_k, sorted_token_ids_offset,
        sorted_token_lens, top_k, max_n_len_offset, bidx);

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++)
    {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / MFMA_M); min_tile_m++)
        {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++)
            {
#pragma unroll
                for (int it = 0; it < 2; it++)
                {

                    const int tile_idx = min_tile_m * (WARP_N / MFMA_N) + min_tile_n * 2 + it;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++)
                    {
                        tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] =
                            C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id];
                        if constexpr (!GroupwisePost)
                        {
                            tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + it] *= b_scale[n_loop][min_tile_n * 2 + it];
                        }
                    }
                }
            }
        }
    }
}

#endif
