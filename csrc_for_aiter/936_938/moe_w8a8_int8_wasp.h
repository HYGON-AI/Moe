// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
// Adapted from deepgemm 68e48ca BM256/BN256/BK128 WASP contiguous INT8.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>
#include <cstdint>
#include <type_traits>

namespace moe_c_w8a8_wasp {

using bhalf_t   = __hip_bfloat16;

template<bool FirstStage, bool CkSorting>
struct MoeParams {
    void* __restrict__ a_ptr;
    void* __restrict__ b_ptr;
    void* __restrict__ d_ptr;
    void* __restrict__ scale_a_ptr;
    void* __restrict__ scale_b_ptr;
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ expert_ids;
    const int32_t* __restrict__ num_tokens_post_pad;
    const float* __restrict__ topk_weights;
    int original_tokens;
    int real_topk;
    int m;
    int n;
    int k;
    int a_stride;
    int d_stride;
    float alpha;
};

// Both sorting paths already pad every expert to the selected BlockM.
template<bool FirstStage, bool CkSorting>
__device__ __forceinline__ int routed_row(
    const MoeParams<FirstStage, CkSorting>& p, int sorted_row) {
    const uint32_t id = static_cast<uint32_t>(p.sorted_token_ids[sorted_row]);
    if constexpr (CkSorting) {
        const uint32_t token = id & 0xffffffu;
        const uint32_t slot = id >> 24;
        return token < p.original_tokens && slot < p.real_topk
            ? static_cast<int>(token * p.real_topk + slot) : -1;
    } else {
        return id < static_cast<uint32_t>(p.original_tokens * p.real_topk)
            ? static_cast<int>(id) : -1;
    }
}

using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using vec4_int8 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;
using vec8_int8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using vec4_int  = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using vec2_bf16 = __attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short;

union union_vec16_i8 {
    int8_t   i8[16];
    vec8_int8 i8x8[2];
    vec4_int8 i8x4[4];
    vec4_fp32 f32x4;
    vec4_int  i32x4;
    int32_t   i32[4];
};


template<int BlockM>
struct GemmTraitsPC {
    using Element      = int8_t;

    static_assert(BlockM == 128 || BlockM == 256);
    static constexpr int kBlockM = BlockM;
    static constexpr int kBlockN = 256;
    static constexpr int kStageK = 128;

    static constexpr int kWaveN  = 32;

    static constexpr int kNConsumerWaves = 8;
    static constexpr int kNProducerWaves = 4;
    static constexpr int kNWaves         = kNConsumerWaves + kNProducerWaves;
    static constexpr int kNThreads       = kNWaves * 64;

    static constexpr int kProdSlabM = kBlockM / kNProducerWaves;


    static constexpr int STAGES           = 2;
    static constexpr size_t A_slab_bytes  = kProdSlabM * kStageK;
    static constexpr size_t A_stage_bytes = kNProducerWaves * A_slab_bytes;
    static constexpr int    kSmemSize     = STAGES * A_stage_bytes;
};


__device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

namespace flash {

__forceinline__ __device__ void wait_all_warp_arrived() {
    __builtin_hcu_sched_barrier(0);
    asm volatile("s_barrier\n");
    __builtin_hcu_sched_barrier(0);
}

template<bool Sync>
__forceinline__ __device__ void wait_all_buffer_data_arrived(int wait_count = 0) {
    __builtin_hcu_sched_barrier(0);
    if constexpr (Sync) {
        asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier\n" :: "n"(wait_count));
    } else {
        asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"(wait_count));
    }
    __builtin_hcu_sched_barrier(0);
}

template<bool Sync>
__forceinline__ __device__ void wait_lds_data_arrived(int wait_count = 0) {
    __builtin_hcu_sched_barrier(0);
    if constexpr (Sync) {
        asm volatile("s_waitcnt lgkmcnt(%0)\n\ts_barrier\n" :: "n"(wait_count));
    } else {
        asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "n"(wait_count));
    }
    __builtin_hcu_sched_barrier(0);
}


}

__device__ __forceinline__ vec4_int mmac_4interleave_i8(
    const vec8_int8& v1, const vec8_int8& v2, const vec4_int& v3
) {
    return __builtin_hcu_mmac_i32_16x16x32_i8(v1, v2, v3);
}

template<bool FastRne = false>
__host__ __device__ __forceinline__ vec2_bf16 DownCastPair_f32_bf16(const vec2_fp32& source) {
#if defined(__gfx92a__) || defined(__gfx938__)
    auto result = __builtin_hcu_cvt_pk_bf16_f32(source[0], source[1], false);
    return *(vec2_bf16*)(&result);
#else
#if defined(__gfx936__)
    if constexpr (FastRne) {
        const auto cvt = [](float value) {
            const uint32_t bits = __builtin_bit_cast(uint32_t, value);
            const uint32_t rounded = bits + 0x7fffU + ((bits >> 16) & 1U);
            // Match HIP rounding and NaN payloads without divergent branches.
            const uint32_t special = bits | (uint32_t((bits & 0xffffU) != 0) << 16);
            const uint32_t mask = -uint32_t((bits & 0x7f800000U) == 0x7f800000U);
            return ((rounded & ~mask) | (special & mask)) >> 16;
        };
        const uint32_t packed = cvt(source[0]) | (cvt(source[1]) << 16);
        return __builtin_bit_cast(vec2_bf16, packed);
    }
#endif
    const __hip_bfloat16 lo_bf16(source[0]);
    const __hip_bfloat16 hi_bf16(source[1]);
    const __hip_bfloat16_raw lo_raw = lo_bf16;
    const __hip_bfloat16_raw hi_raw = hi_bf16;
    uint32_t packed = static_cast<uint32_t>(lo_raw.x) |
                      (static_cast<uint32_t>(hi_raw.x) << 16);
    return *(vec2_bf16*)(&packed);
#endif
}


typedef int32_t dword2_t __attribute__((ext_vector_type(2)));
typedef int32_t dword4_t __attribute__((ext_vector_type(4)));
typedef int32_t buffer_rsrc_t __attribute__((ext_vector_type(4)));

template <bool cache_swizzle_enable = true>
__device__ __forceinline__ buffer_rsrc_t make_buffer_rsrc(const void* ptr, int stride, uint32_t size_bytes) {
    buffer_rsrc_t r;
    uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    r[0] = (int32_t)(addr & 0xffffffff);
    r[1] = (int32_t)(addr >> 32);
    r[2] = (int32_t)size_bytes;
    r[3] = 0x00020000;

    if (cache_swizzle_enable && (((stride ^ (stride - 1)) == 0) || ((stride & 128) == 0))) {
        uint32_t val = 1u << 30;
        val |= (stride & 0x3fff) << 16;
        r[1] |= val;
    }

    return r;
}


__device__ __forceinline__ void buffer_load_dwordx4_x8_to_lds(
    buffer_rsrc_t rsrc,
    uint32_t voff0, uint32_t voff1, uint32_t voff2, uint32_t voff3,
    uint32_t voff4, uint32_t voff5, uint32_t voff6, uint32_t voff7,
    uint32_t lds_m0, uint32_t s_col_base) {
    asm volatile(
        "s_mov_b32 m0, %8\n\t"
        "buffer_load_dwordx4 %0, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %1, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %2, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %3, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %4, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %5, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %6, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %7, %9, %10 offen lds\n"
        :
        : "v"(voff0), "v"(voff1), "v"(voff2), "v"(voff3),
          "v"(voff4), "v"(voff5), "v"(voff6), "v"(voff7),
          "s"(lds_m0), "s"(rsrc), "s"(s_col_base)
        : "memory", "m0");
}


__device__ __forceinline__ void buffer_load_dwordx4_x8_to_lds_soff(
    buffer_rsrc_t rsrc,
    uint32_t voff,
    uint32_t lds_m0,
    uint32_t s_soff0, uint32_t s_soff1, uint32_t s_soff2, uint32_t s_soff3,
    uint32_t s_soff4, uint32_t s_soff5, uint32_t s_soff6, uint32_t s_soff7) {
    asm volatile(
        "s_mov_b32 m0, %1\n\t"
        "buffer_load_dwordx4 %0, %2, %3 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %4 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %5 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %6 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %7 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %8 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %9 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %0, %2, %10 offen lds\n"
        :
        : "v"(voff), "s"(lds_m0), "s"(rsrc),
          "s"(s_soff0), "s"(s_soff1), "s"(s_soff2), "s"(s_soff3),
          "s"(s_soff4), "s"(s_soff5), "s"(s_soff6), "s"(s_soff7)
        : "memory", "m0");
}


__forceinline__ __device__ void buffer_load_i32x4_pair32(
    const char* base, dword4_t& value, int byte_offset) {
    __builtin_hcu_sched_barrier(0);
    using intx4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;
    intx4 resource;
    *reinterpret_cast<uint64_t*>(&resource) = reinterpret_cast<uint64_t>(base);
    resource[1] += 0x40800000;
    resource[2] = 0x80000000;
    resource[3] = 0x00020000;
    asm volatile(
        "s_nop 2\n\t"
        "buffer_load_dwordx4 %0,%1,%2,0, offen offset:0\n"
        : "=v"(value), "+v"(byte_offset), "+s"(resource)
        :
        : "memory");
}

__forceinline__ __device__ void buffer_load_i32x4_pair32(
    buffer_rsrc_t resource, dword4_t& value, int byte_offset, int scalar_offset) {
    __builtin_hcu_sched_barrier(0);
    asm volatile(
        "s_nop 2\n\t"
        "buffer_load_dwordx4 %0,%1,%2,%3 offen offset:0\n"
        : "=&v"(value)
        : "v"(byte_offset), "s"(resource), "s"(scalar_offset)
        : "memory");
}


template<int EXTRA = 0>
__device__ __forceinline__ void ds_read_b128_x16_imm(
    uint32_t base,
    dword4_t& o0,  dword4_t& o1,  dword4_t& o2,  dword4_t& o3,
    dword4_t& o4,  dword4_t& o5,  dword4_t& o6,  dword4_t& o7,
    dword4_t& o8,  dword4_t& o9,  dword4_t& o10, dword4_t& o11,
    dword4_t& o12, dword4_t& o13, dword4_t& o14, dword4_t& o15) {
    static_assert(EXTRA >= 0 && EXTRA + 30720 <= 65535,
                  "ds_read_b128 imm offset overflows 16-bit unsigned field");
    asm volatile(
        "ds_read_b128 %0,  %16 offset:%c17\n\t"
        "ds_read_b128 %1,  %16 offset:%c18\n\t"
        "ds_read_b128 %2,  %16 offset:%c19\n\t"
        "ds_read_b128 %3,  %16 offset:%c20\n\t"
        "ds_read_b128 %4,  %16 offset:%c21\n\t"
        "ds_read_b128 %5,  %16 offset:%c22\n\t"
        "ds_read_b128 %6,  %16 offset:%c23\n\t"
        "ds_read_b128 %7,  %16 offset:%c24\n\t"
        "ds_read_b128 %8,  %16 offset:%c25\n\t"
        "ds_read_b128 %9,  %16 offset:%c26\n\t"
        "ds_read_b128 %10, %16 offset:%c27\n\t"
        "ds_read_b128 %11, %16 offset:%c28\n\t"
        "ds_read_b128 %12, %16 offset:%c29\n\t"
        "ds_read_b128 %13, %16 offset:%c30\n\t"
        "ds_read_b128 %14, %16 offset:%c31\n\t"
        "ds_read_b128 %15, %16 offset:%c32\n"
        : "=&v"(o0),  "=&v"(o1),  "=&v"(o2),  "=&v"(o3),
          "=&v"(o4),  "=&v"(o5),  "=&v"(o6),  "=&v"(o7),
          "=&v"(o8),  "=&v"(o9),  "=&v"(o10), "=&v"(o11),
          "=&v"(o12), "=&v"(o13), "=&v"(o14), "=&v"(o15)
        : "v"(base),
          "n"(EXTRA + 0),     "n"(EXTRA + 2048),  "n"(EXTRA + 4096),  "n"(EXTRA + 6144),
          "n"(EXTRA + 8192),  "n"(EXTRA + 10240), "n"(EXTRA + 12288), "n"(EXTRA + 14336),
          "n"(EXTRA + 16384), "n"(EXTRA + 18432), "n"(EXTRA + 20480), "n"(EXTRA + 22528),
          "n"(EXTRA + 24576), "n"(EXTRA + 26624), "n"(EXTRA + 28672), "n"(EXTRA + 30720)
        : "memory");
}


template<int EXTRA = 0>
__device__ __forceinline__ void ds_read_b128_x8_imm(
    uint32_t base, dword4_t& o0, dword4_t& o1, dword4_t& o2, dword4_t& o3,
    dword4_t& o4, dword4_t& o5, dword4_t& o6, dword4_t& o7) {
    asm volatile(
        "ds_read_b128 %0, %8 offset:%c9\n\t"
        "ds_read_b128 %1, %8 offset:%c10\n\t"
        "ds_read_b128 %2, %8 offset:%c11\n\t"
        "ds_read_b128 %3, %8 offset:%c12\n\t"
        "ds_read_b128 %4, %8 offset:%c13\n\t"
        "ds_read_b128 %5, %8 offset:%c14\n\t"
        "ds_read_b128 %6, %8 offset:%c15\n\t"
        "ds_read_b128 %7, %8 offset:%c16\n"
        : "=&v"(o0), "=&v"(o1), "=&v"(o2), "=&v"(o3),
          "=&v"(o4), "=&v"(o5), "=&v"(o6), "=&v"(o7)
        : "v"(base), "n"(EXTRA), "n"(EXTRA + 2048), "n"(EXTRA + 4096),
          "n"(EXTRA + 6144), "n"(EXTRA + 8192), "n"(EXTRA + 10240),
          "n"(EXTRA + 12288), "n"(EXTRA + 14336)
        : "memory");
}

__device__ __forceinline__ void buffer_load_dwordx4_x4_to_lds(
    buffer_rsrc_t rsrc, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
    uint32_t lds_m0, uint32_t soff) {
    asm volatile(
        "s_mov_b32 m0, %4\n\t"
        "buffer_load_dwordx4 %0, %5, %6 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %1, %5, %6 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %2, %5, %6 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %3, %5, %6 offen lds\n"
        :: "v"(v0), "v"(v1), "v"(v2), "v"(v3),
           "s"(lds_m0), "s"(rsrc), "s"(soff) : "memory", "m0");
}

template<int EXTRA = 0>
__device__ __forceinline__ void ds_read_b128_x4_imm(
    uint32_t base, dword4_t& o0, dword4_t& o1, dword4_t& o2, dword4_t& o3) {
    asm volatile(
        "ds_read_b128 %0, %4 offset:%c5\n\t"
        "ds_read_b128 %1, %4 offset:%c6\n\t"
        "ds_read_b128 %2, %4 offset:%c7\n\t"
        "ds_read_b128 %3, %4 offset:%c8\n"
        : "=&v"(o0), "=&v"(o1), "=&v"(o2), "=&v"(o3)
        : "v"(base), "n"(EXTRA), "n"(EXTRA + 2048),
          "n"(EXTRA + 4096), "n"(EXTRA + 6144) : "memory");
}

union union_vec16_dword {
    dword4_t   d4[4];
    vec8_int8  v8[8];
    float      f[16];
};


template<typename Element_O, bool FastRne = false, int WavesM = 1, bool FastStore = false, int BlockM = 256>
__device__ __forceinline__ void pair32_writeback_bf16(
    Element_O* output,
    int ldd,
    int col_base,
    int consumer_id,
    int lane_id,
    const vec4_int* acc_o,
    const float* scale_lds,
    float alpha) {
    static_assert(sizeof(Element_O) == 2,
                  "pair32 WASP currently writes a 2-byte output");
    const int row_id = lane_id & 15;
    const int col_id = lane_id >> 4;
    constexpr int kWaveM = BlockM / WavesM;
    constexpr int kWaveN = 32 * WavesM;
    const int row_start = consumer_id / (8 / WavesM) * kWaveM;
    const int col_start = consumer_id % (8 / WavesM) * kWaveN + row_id * 2;
    vec2_fp32 col_scale[WavesM];
#pragma unroll
    for (int nt = 0; nt < WavesM; ++nt) {
        col_scale[nt] = *reinterpret_cast<const vec2_fp32*>(
            scale_lds + BlockM + col_start + nt * 32);
    }
    buffer_rsrc_t output_rsrc;
    if constexpr (FastStore) {
        output_rsrc = make_buffer_rsrc<false>(output, 0, 0x80000000u);
    }
    constexpr int kBatchTilesM = WavesM == 1 ? 4 : 2;
    // Use the FP8 N-pair layout, but skip padded rows before conversion.
#pragma unroll
    for (int m_base = 0; m_base < kWaveM / 16; m_base += kBatchTilesM) {
        float row_scale[kBatchTilesM][4];
        int row[kBatchTilesM][4];
#pragma unroll
        for (int m = 0; m < kBatchTilesM; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int sorted_row = row_start + (m_base + m) * 16 + r * 4 + col_id;
                row_scale[m][r] = scale_lds[sorted_row];
                row[m][r] = reinterpret_cast<const int32_t*>(
                    scale_lds + BlockM + 256)[sorted_row];
            }
        }
        // Batch LDS and routing reads before consuming them in the epilogue.
        asm volatile("" ::: "memory");
#pragma unroll
        for (int m = 0; m < kBatchTilesM; ++m) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (row[m][r] >= 0) {
                    const float row_alpha = row_scale[m][r] * alpha;
#pragma unroll
                    for (int nt = 0; nt < WavesM; ++nt) {
                        const int acc = (m_base + m) * 2 * WavesM + nt * 2;
                        vec2_fp32 scaled = {static_cast<float>(acc_o[acc][r]),
                                           static_cast<float>(acc_o[acc + 1][r])};
                        scaled = scaled * row_alpha;
                        scaled = scaled * col_scale[nt];
                        const vec2_bf16 value = DownCastPair_f32_bf16<FastRne>(scaled);
                        const int col = col_base + col_start + nt * 32;
                        if constexpr (FastStore) {
                            const uint32_t offset = (row[m][r] * ldd + col) * sizeof(Element_O);
                            const uint32_t packed = __builtin_bit_cast(uint32_t, value);
                            asm volatile("buffer_store_dword %0, %1, %2, 0 offen\n"
                                :: "v"(packed), "v"(offset), "s"(output_rsrc) : "memory");
                        } else {
                            *reinterpret_cast<vec2_bf16*>(
                                output + static_cast<int64_t>(row[m][r]) * ldd + col) = value;
                        }
                    }
                }
            }
        }
    }
}


template<bool FirstStage, bool CkSorting, int BlockM>
__device__ __forceinline__ void producer_body(
    const MoeParams<FirstStage, CkSorting>& params,
    int prod_id,
    int8_t* a_lds_all,
    int by, int bx,
    int lane_id
) {
    using Traits = GemmTraitsPC<BlockM>;
    constexpr int kBlockM    = Traits::kBlockM;
    constexpr int kBlockN    = Traits::kBlockN;
    constexpr int kStageK    = Traits::kStageK;
    constexpr int kProdSlabM = Traits::kProdSlabM;

    const int K   = params.k;
    const int lda = params.a_stride;
    const int a_row_start = by * kBlockM + prod_id * kProdSlabM;

    buffer_rsrc_t a_rsrc = make_buffer_rsrc(
        reinterpret_cast<int8_t*>(params.a_ptr), lda,
        (uint32_t)(params.m * lda));

    const int n_stages = ceil_div(K, kStageK);
    const int row_group = lane_id >> 3;
    const int k_in_panel = ((lane_id & 7) ^ row_group) * 16;

    constexpr int kRowLoads = kProdSlabM / 8;
    uint32_t row_offsets[kRowLoads];
#pragma unroll
    for (int i = 0; i < kRowLoads; ++i) {
        const int route = routed_row(params, a_row_start + row_group + i * 8);
        const int input_row = FirstStage ? route / params.real_topk : route;
        row_offsets[i] = route >= 0
            ? static_cast<uint32_t>(input_row * lda + k_in_panel)
            : 0x80000000u;
    }

    constexpr uint32_t A_stage_bytes_ct = (uint32_t)Traits::A_stage_bytes;
    const uint32_t prod_slab_off = (uint32_t)(prod_id * (int)Traits::A_slab_bytes);

    const int32_t expert_id =
        __builtin_hcu_readfirstlane(params.expert_ids[by]);
    const uint32_t scale_lds_base =
        (uint32_t)((n_stages & 1) * (int)Traits::A_stage_bytes);
    const uint32_t tid4 = ((uint32_t)lane_id << 2) + (uint32_t)(prod_id * 256);
    const int scale_route = lane_id < kProdSlabM
        ? routed_row(params, a_row_start + lane_id) : -1;
    const int scale_row = FirstStage ? scale_route / params.real_topk : scale_route;
    const uint32_t ga = scale_route >= 0 ? static_cast<uint32_t>(scale_row * 4) : 0x80000000u;
    const uint32_t gb = (uint32_t)(bx * kBlockN * 4) + tid4;
    uint32_t la = 0;

    asm volatile("v_add_u32_e32 %0, %1, %2" : "=v"(la) : "v"(tid4), "v"(scale_lds_base));
    buffer_rsrc_t sa_rsrc = make_buffer_rsrc<false>(
        reinterpret_cast<const float*>(params.scale_a_ptr), 4, (uint32_t)(params.m * 4));
    buffer_rsrc_t sb_rsrc = make_buffer_rsrc<false>(
        reinterpret_cast<const float*>(params.scale_b_ptr)
            + static_cast<int64_t>(expert_id) * params.n,
        4, (uint32_t)(params.n * 4));

    const int k_last_stage = n_stages - 1;
    auto issue_stage = [&](int k_stage, int stage) {
        const uint32_t s_col_base = static_cast<uint32_t>(k_stage * kStageK);
        const uint32_t lds_m0 = prod_slab_off + stage * A_stage_bytes_ct;
        if constexpr (kRowLoads == 8) {
            buffer_load_dwordx4_x8_to_lds(
                a_rsrc, row_offsets[0], row_offsets[1], row_offsets[2], row_offsets[3],
                row_offsets[4], row_offsets[5], row_offsets[6], row_offsets[7],
                lds_m0, s_col_base);
        } else {
            buffer_load_dwordx4_x4_to_lds(
                a_rsrc, row_offsets[0], row_offsets[1], row_offsets[2], row_offsets[3],
                lds_m0, s_col_base);
        }
    };

    issue_stage(0, 0);
    flash::wait_all_warp_arrived();
    flash::wait_all_buffer_data_arrived<false>(0);
    flash::wait_all_warp_arrived();


    #pragma unroll 1
    for (int k_stage = 0; k_stage + 2 < n_stages; ++k_stage) {
        const int stage_to_issue = k_stage + 1;
        const int s = stage_to_issue & 1;
        issue_stage(stage_to_issue, s);
        flash::wait_all_warp_arrived();
        flash::wait_all_buffer_data_arrived<false>(0);
        flash::wait_all_warp_arrived();
    }


    if (n_stages >= 2) {
        const int s = k_last_stage & 1;
        issue_stage(k_last_stage, s);
        flash::wait_all_warp_arrived();
        flash::wait_all_buffer_data_arrived<false>(0);
        flash::wait_all_warp_arrived();
    }

    flash::wait_all_warp_arrived();
    float sa, sb;
    const uint32_t zero = 0;
    asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:0\n\t"
                 : "=&v"(sa) : "v"(ga), "s"(sa_rsrc), "s"(zero) : "memory");
    asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:0\n\t"
                 : "=&v"(sb) : "v"(gb), "s"(sb_rsrc), "s"(zero) : "memory");
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    if constexpr (!FirstStage) {
        if (scale_route >= 0) sa *= params.topk_weights[scale_route];
    }
    if (lane_id < kProdSlabM) {
        const uint32_t row_la = scale_lds_base + (prod_id * kProdSlabM + lane_id) * 4;
        asm volatile("ds_write_b32 %0, %1\n\t" :: "v"(row_la), "v"(sa) : "memory");
        asm volatile("ds_write_b32 %0, %1 offset:%c2\n\t"
            :: "v"(row_la), "v"(scale_route), "n"((BlockM + 256) * 4) : "memory");
    }
    // Reuse the opposite A stage for row scales, column scales and routes.
    asm volatile("ds_write_b32 %0, %1 offset:%c2\n\t"
        :: "v"(la), "v"(sb), "n"(BlockM * 4) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
    flash::wait_all_warp_arrived();
    flash::wait_all_warp_arrived();
}


template<bool FirstStage, bool CkSorting, typename Element_O, bool FastShape = false, int WavesM = 1, bool FastOutput = false, int BlockM = 256>
__device__ __forceinline__ void consumer_body(
    const MoeParams<FirstStage, CkSorting>& params,
    int consumer_id,
    int8_t* a_lds_all,
    int by, int bx,
    int lane_id
) {
    using Traits = GemmTraitsPC<BlockM>;
    constexpr int kBlockM     = Traits::kBlockM;
    constexpr int kBlockN     = Traits::kBlockN;
    constexpr int kStageK     = Traits::kStageK;
    static_assert(WavesM == 1 || WavesM == 2);
    constexpr int kWaveN      = 32 * WavesM;
    constexpr int kMmacM      = 16;
    constexpr int kMmacN      = 16;
    constexpr int kSubTilesM  = kBlockM / WavesM / kMmacM;
    constexpr int kSubTilesN  = kWaveN / kMmacN;

    const int row_in_tile = lane_id & 15;
    const int col_group   = lane_id >> 4;
    const int n_stages = ceil_div(params.k, kStageK);

    const int M = params.m, N = params.n;
    const int ldd = params.d_stride;
    const int b_col_start = bx * kBlockN;
    const int32_t expert_id =
        __builtin_hcu_readfirstlane(params.expert_ids[by]);
    const int b_n16_tiles = N / 16;
    const char* expert_weight = reinterpret_cast<const char*>(params.b_ptr)
        + static_cast<int64_t>(expert_id) * N * params.k;


    vec4_int acc_o[kSubTilesM * kSubTilesN] = {};


    union_vec16_i8 a_regs[kSubTilesM];


    union_vec16_dword b_regs[2][WavesM];


    const int r7 = row_in_tile & 7;
    const uint32_t a_base_h0 = (uint32_t)(
        reinterpret_cast<size_t>(a_lds_all)
        + (consumer_id / (8 / WavesM) * (BlockM / WavesM) + row_in_tile) * kStageK
        + (col_group ^ r7) * 16);
    const uint32_t a_base_h1 = (uint32_t)(
        reinterpret_cast<size_t>(a_lds_all)
        + (consumer_id / (8 / WavesM) * (BlockM / WavesM) + row_in_tile) * kStageK
        + ((col_group + 4) ^ r7) * 16);

    auto ds_read_a_half = [&]<int STAGE_OFF>(uint32_t base) {
        if constexpr (kSubTilesM == 16) {
            ds_read_b128_x16_imm<STAGE_OFF>(base,
                reinterpret_cast<dword4_t&>(a_regs[0].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[1].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[2].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[3].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[4].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[5].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[6].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[7].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[8].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[9].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[10].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[11].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[12].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[13].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[14].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[15].i32x4));
        } else if constexpr (kSubTilesM == 8) {
            ds_read_b128_x8_imm<STAGE_OFF>(base,
                reinterpret_cast<dword4_t&>(a_regs[0].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[1].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[2].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[3].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[4].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[5].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[6].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[7].i32x4));
        } else {
            ds_read_b128_x4_imm<STAGE_OFF>(base,
                reinterpret_cast<dword4_t&>(a_regs[0].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[1].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[2].i32x4),
                reinterpret_cast<dword4_t&>(a_regs[3].i32x4));
        }
    };


    auto mmac_half = [&]<int BUF>(int k_half) {
        const int b_base = k_half * 4;
#pragma unroll
        for (int m = 0; m < kSubTilesM; ++m) {
#pragma unroll
            for (int n = 0; n < WavesM; ++n) {
                const int acc = m * kSubTilesN + n * 2;
                acc_o[acc] = mmac_4interleave_i8(
                    b_regs[BUF][n].v8[b_base], a_regs[m].i8x8[0], acc_o[acc]);
                acc_o[acc] = mmac_4interleave_i8(
                    b_regs[BUF][n].v8[b_base + 1], a_regs[m].i8x8[1], acc_o[acc]);
                acc_o[acc + 1] = mmac_4interleave_i8(
                    b_regs[BUF][n].v8[b_base + 2], a_regs[m].i8x8[0], acc_o[acc + 1]);
                acc_o[acc + 1] = mmac_4interleave_i8(
                    b_regs[BUF][n].v8[b_base + 3], a_regs[m].i8x8[1], acc_o[acc + 1]);
            }
        }
    };


    // Keep the per-lane Marlin offset fixed and advance K with a scalar offset.
#if defined(__gfx936__)
    constexpr bool kFastShape = FastShape;
    constexpr bool kFastRne = kFastShape || WavesM == 2;
#else
    constexpr bool kFastShape = false;
    constexpr bool kFastRne = false;
#endif
    constexpr bool kScalarB = kFastShape || WavesM == 2;
    buffer_rsrc_t b_rsrc = make_buffer_rsrc<false>(expert_weight, 0, 0x80000000u);
    b_rsrc[1] += 0x40800000;
    const int kg = lane_id >> 4;
    const int n0 = bx * kBlockN + consumer_id % (8 / WavesM) * kWaveN + row_in_tile * 2;
    const int b_offset = (((n0 / 16) * 4 + kg) * 16 + n0 % 16) * 16;
    auto buffer_load_b = [&]<int BUF>(int stage_id) {
#pragma unroll
        for (int nt = 0; nt < WavesM; ++nt) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    if constexpr (kScalarB) {
                        buffer_load_i32x4_pair32(
                            b_rsrc, b_regs[BUF][nt].d4[half * 2 + pair],
                            b_offset + nt * 2048 + pair * 16,
                            stage_id * 128 * N + half * 64 * N);
                    } else {
                        const int n = n0 + nt * 32 + pair;
                        const int64_t k64 = stage_id * 2 + half;
                        const int64_t offset = (((k64 * b_n16_tiles + n / 16) * 4 + kg)
                                              * 16 + n % 16) * 16;
                        buffer_load_i32x4_pair32(
                            expert_weight,
                            b_regs[BUF][nt].d4[half * 2 + pair], static_cast<int>(offset));
                    }
                }
            }
        }
    };


    auto stage_body_C = [&]<int CUR>(int k_stage) {
        constexpr int NXT = CUR ^ 1;
        constexpr int STAGE_OFF = CUR * (int)Traits::A_stage_bytes;

        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h0);

        flash::wait_lds_data_arrived<false>(0);
        flash::wait_all_buffer_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(0);
        flash::wait_all_warp_arrived();


        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h1);
        if (k_stage + 1 < n_stages) {
            buffer_load_b.template operator()<NXT>(k_stage + 1);
        }

        flash::wait_lds_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(1);
        flash::wait_all_warp_arrived();
    };


    auto stage_body_Cp = [&]<int CUR, bool IS_FIRST>(int k_stage) {
        constexpr int NXT = CUR ^ 1;
        constexpr int STAGE_OFF = CUR * (int)Traits::A_stage_bytes;

        if constexpr (!IS_FIRST) {
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<NXT>(1);
        }

        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h0);
        flash::wait_all_warp_arrived();


        flash::wait_lds_data_arrived<false>(0);
        flash::wait_all_buffer_data_arrived<false>(0);
        mmac_half.template operator()<CUR>(0);

        ds_read_a_half.template operator()<STAGE_OFF>(a_base_h1);
        if (k_stage + 1 < n_stages) {
            buffer_load_b.template operator()<NXT>(k_stage + 1);
        }
        flash::wait_all_warp_arrived();
    };


    buffer_load_b.template operator()<0>(0);
    flash::wait_all_warp_arrived();

    flash::wait_all_buffer_data_arrived<false>(0);
    flash::wait_all_warp_arrived();

    const bool is_C_prime = WavesM == 1 ? consumer_id >= 4 : (consumer_id & 1) != 0;


    if (!is_C_prime) {
        #pragma unroll 1
        for (int k = 0; k < n_stages; k += 2) {
            stage_body_C.template operator()<0>(k);
            if (k + 1 < n_stages) {
                stage_body_C.template operator()<1>(k + 1);
            }
        }
    } else {


        if ((n_stages & 1) != 0) {

            stage_body_Cp.template operator()<0, true>(0);
            #pragma unroll 1
            for (int k = 1; k < n_stages - 1; k += 2) {
                stage_body_Cp.template operator()<1, false>(k);
                stage_body_Cp.template operator()<0, false>(k + 1);
            }
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<0>(1);
        } else {


            stage_body_Cp.template operator()<0, true>(0);
            #pragma unroll 1
            for (int k = 1; k < n_stages - 1; k += 2) {
                stage_body_Cp.template operator()<1, false>(k);
                stage_body_Cp.template operator()<0, false>(k + 1);
            }
            stage_body_Cp.template operator()<1, false>(n_stages - 1);
            flash::wait_lds_data_arrived<false>(0);
            mmac_half.template operator()<1>(1);
        }
    }

    const uint32_t scale_lds_base =
        (uint32_t)((n_stages & 1) * (int)Traits::A_stage_bytes);
    flash::wait_all_warp_arrived();


    Element_O* d_ptr = reinterpret_cast<Element_O*>(params.d_ptr);
    const int writeback_lane_id = threadIdx.x & 63;
    pair32_writeback_bf16<Element_O, kFastRne, WavesM, kFastShape || FastOutput, BlockM>(
        d_ptr, ldd,
        b_col_start,
        consumer_id, writeback_lane_id,
        acc_o, reinterpret_cast<const float*>(a_lds_all + scale_lds_base), params.alpha);
}


template<bool FirstStage, bool CkSorting, typename Element_O, bool FastShape = false, int WavesM = 1, bool FastOutput = false, int BlockM = 256>
__device__ __forceinline__ void compute_int8_pair32_wasp(
    const MoeParams<FirstStage, CkSorting>& params) {
    using Traits = GemmTraitsPC<BlockM>;
    using Element = typename Traits::Element;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int warp_id = __builtin_hcu_readfirstlane(threadIdx.x / 64);
    const int lane_id = threadIdx.x & 63;

    extern __shared__ Element smem[];
    if (warp_id < Traits::kNProducerWaves) {
        producer_body<FirstStage, CkSorting, BlockM>(params, warp_id, smem, by, bx, lane_id);
    } else {
        const int physical = warp_id - Traits::kNProducerWaves;
        const int consumer = WavesM == 1 ? physical : (physical % 4 * 2 + physical / 4);
        consumer_body<FirstStage, CkSorting, Element_O, FastShape, WavesM, FastOutput, BlockM>(
            params, consumer, smem, by, bx, lane_id);
    }
}

template<bool FirstStage, bool CkSorting, typename Element_O, bool FastShape = false, int WavesM = 1, bool FastOutput = false, int BlockM = 256>
__global__ __launch_bounds__(GemmTraitsPC<BlockM>::kNThreads, 1)
void int8_pair32_wasp_kernel(MoeParams<FirstStage, CkSorting> params) {
    if constexpr (FastShape) {
        params.original_tokens = 8192;
        params.m = FirstStage ? 8192 : 65536;
        params.real_topk = 8;
        params.n = params.d_stride = FirstStage ? 512 : 2048;
        params.k = params.a_stride = FirstStage ? 2048 : 256;
        params.alpha = 1.0f;
    }
    const int sorted_start = blockIdx.y * BlockM;
    // Block-uniform guards must precede all producer/consumer barriers.
    if (sorted_start >= params.num_tokens_post_pad[0]) {
        return;
    }
    if (params.expert_ids[blockIdx.y] < 0) {
        // EP routes still occupy output slots even when their expert is remote.
        auto* output = reinterpret_cast<Element_O*>(params.d_ptr);
        for (int i = threadIdx.x * 4; i < BlockM * 256;
             i += GemmTraitsPC<BlockM>::kNThreads * 4) {
            const int route = routed_row(params, sorted_start + i / 256);
            if (route >= 0) {
                const int64_t offset = static_cast<int64_t>(route) * params.d_stride
                    + blockIdx.x * 256 + i % 256;
                *reinterpret_cast<uint64_t*>(output + offset) = 0;
            }
        }
        return;
    }
    compute_int8_pair32_wasp<FirstStage, CkSorting, Element_O, FastShape, WavesM, FastOutput, BlockM>(params);
}

} // namespace moe_c_w8a8_wasp

template<bool FirstStage, bool CkSorting, typename OutT, int WavesM, int BlockM>
void launch_moe_w8a8_int8_wasp(const GemmParams<char, OutT>& p) {
    using namespace moe_c_w8a8_wasp;
    using Traits = GemmTraitsPC<BlockM>;
    TORCH_CHECK((std::is_same_v<OutT, bhalf_t>),
                "W8A8 WASP requires BF16 output");
    TORCH_CHECK(!p.tensorwise_scale && p.is_marlin,
                "W8A8 WASP requires channelwise Marlin INT8 weights");
    TORCH_CHECK(p.size_n > 0 && p.size_n % 256 == 0 &&
                p.size_k >= 128 && p.size_k % 128 == 0,
                "W8A8 WASP requires N % 256 == 0 and K % 128 == 0");
    TORCH_CHECK(p.real_size_k == p.size_k && p.stride_asm == 1 &&
                p.stride_bsn == 1 && p.stride_bse == p.size_n,
                "W8A8 WASP requires contiguous input and channel scales");
    TORCH_CHECK(p.real_topk > 0 &&
                (FirstStage || p.size_m % p.real_topk == 0),
                "invalid W8A8 WASP topk or routed input rows");
    if constexpr (CkSorting) {
        TORCH_CHECK(p.real_topk < 256,
                    "CK sorting requires topk < 256 for its 8-bit slot encoding");
    }
    TORCH_CHECK(FirstStage || p.topk_weights,
                "W8A8 WASP GEMM2 requires topk weights");
    TORCH_CHECK(static_cast<uint64_t>(p.size_m) * p.size_k < 0x80000000ull,
                "W8A8 WASP input exceeds the 2 GiB buffer addressing range");
    TORCH_CHECK(static_cast<uint64_t>(p.size_n) * p.size_k < 0x80000000ull,
                "W8A8 WASP expert weights exceed the 2 GiB buffer addressing range");
    if (p.size_m == 0 || p.sorted_token_lens == 0) return;

    MoeParams<FirstStage, CkSorting> params{};
    params.a_ptr = const_cast<char*>(p.ptr_A);
    params.b_ptr = const_cast<char*>(p.ptr_B0);
    params.d_ptr = p.ptr_C;
    params.scale_a_ptr = p.ptr_A_scale;
    params.scale_b_ptr = p.ptr_B_scale;
    params.sorted_token_ids = p.sorted_token_ids;
    params.expert_ids = p.expert_ids;
    params.num_tokens_post_pad = p.num_tokens_post_pad_ptr;
    params.topk_weights = FirstStage ? nullptr : p.topk_weights;
    params.original_tokens = FirstStage ? p.size_m : p.size_m / p.real_topk;
    params.real_topk = p.real_topk;
    params.m = p.size_m;
    params.n = p.size_n;
    params.k = p.size_k;
    params.a_stride = p.size_k;
    params.d_stride = p.size_n;
    params.alpha = 1.0f;
    const dim3 grid(p.size_n / 256, (p.sorted_token_lens + BlockM - 1) / BlockM);
    if constexpr (FirstStage) {
        // Specialize only the measured workload; other shapes retain dynamic geometry.
        if (p.size_m == 8192 && p.real_topk == 8 && p.size_n == 512 && p.size_k == 2048) {
            hipLaunchKernelGGL(
                (int8_pair32_wasp_kernel<FirstStage, CkSorting, bhalf_t, true, WavesM, false, BlockM>), grid,
                dim3(Traits::kNThreads), Traits::kSmemSize,
                at::cuda::getCurrentHIPStream(), params);
            return;
        }
    }
    if constexpr (!FirstStage) {
        if (p.size_m == 65536 && p.real_topk == 8 && p.size_n == 2048 && p.size_k == 256) {
            hipLaunchKernelGGL(
                (int8_pair32_wasp_kernel<FirstStage, CkSorting, bhalf_t, true, WavesM, true, BlockM>), grid,
                dim3(Traits::kNThreads), Traits::kSmemSize,
                at::cuda::getCurrentHIPStream(), params);
            return;
        }
    }
    if constexpr (WavesM == 2) {
        const uint64_t output_rows = uint64_t(p.size_m) * (FirstStage ? p.real_topk : 1);
        if (output_rows * p.size_n * sizeof(bhalf_t) < 0x80000000ull) {
            hipLaunchKernelGGL(
                (int8_pair32_wasp_kernel<FirstStage, CkSorting, bhalf_t, false, WavesM, true, BlockM>), grid,
                dim3(Traits::kNThreads), Traits::kSmemSize,
                at::cuda::getCurrentHIPStream(), params);
            return;
        }
    }
    hipLaunchKernelGGL(
        (int8_pair32_wasp_kernel<FirstStage, CkSorting, bhalf_t, false, WavesM, false, BlockM>), grid,
        dim3(Traits::kNThreads), Traits::kSmemSize,
        at::cuda::getCurrentHIPStream(), params);
}
