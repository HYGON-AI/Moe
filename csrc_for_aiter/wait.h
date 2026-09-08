#ifndef WAIT_H
#define WAIT_H

#define USE_PINGPANG_BUFFER

template <const int COUNT>
__forceinline__ __device__ void buffer_load_lds_dwordx1_wait()
{
    asm volatile(
        "s_waitcnt vmcnt(%0)\n\t"
        "s_barrier\n" ::"B"(COUNT)
        :);
}

template <const int COUNT>
__forceinline__ __device__ void buffer_load_lds_dwordx1_wait_nosync()
{
    asm volatile(
        "s_waitcnt vmcnt(%0)\n\t" ::"B"(COUNT)
        :);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
inline __device__ void buffer_load_lds_dwordx1_wait()
{
    asm volatile("s_waitcnt vmcnt(0) \n\t"
                 "s_barrier");
}

__forceinline__ __device__ void s_barrier()
{
    asm volatile("s_barrier\n");
}

#define lgkmcnt_wait(X)                               \
    __builtin_amdgcn_sched_barrier(0);                \
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "I"(X)); \
    __builtin_amdgcn_sched_barrier(0);

#define vmcnt_wait(X)                  \
    __builtin_amdgcn_sched_barrier(0); \
    asm volatile(                      \
        "s_waitcnt vmcnt(%0)\n\t"      \
        "s_barrier\n" ::"I"(X)         \
        :);                            \
    __builtin_amdgcn_sched_barrier(0);

#endif
