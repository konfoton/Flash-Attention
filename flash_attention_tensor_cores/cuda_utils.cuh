#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>


#define ROWS 4
#define COLS 2
#define HEAD_DIM 128
#define BYTES_PER_VEC4_ACCESS 16

 
/*
Total load (64, 128) from global to shared memory
Each warp loads (16, 128) 
but divied into 8 loads of (4, 64) so (4, 2)
*/

__device__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
 
template <int ngroups>
__device__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
}
 

template <int size, typename T>
__device__ void cp_async(T *smem_to, T *gmem_from) {
    static_assert(size == 16);
    
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}




template <typename T>
struct GM2SM_async {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS, T>(smem, gmem);
    }
};
 
template <typename T>
struct SM2GM {
    __device__ constexpr void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};
 
template <typename op,
          typename value_t>
__forceinline__ __device__ constexpr void copy_block_GSM(
	value_t *gmem,
	value_t *smem,
    const int warp_id
){

    if constexpr (std::is_same_v<value_t, __half>) {
    // float16 implementation
    int lane_id = threadIdx.x % 32;
    int thread_row = lane_id / 8; 
    int thread_col = lane_id % 8;
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            op()(&gmem[warp_id * 16 * HEAD_DIM + r * 4 * HEAD_DIM + thread_row * HEAD_DIM + c * 8 * 8 + thread_col * 8],
                 &smem[warp_id * 16 * HEAD_DIM + r * 4 * HEAD_DIM + thread_row * HEAD_DIM + c * 8 * 8 + thread_col * 8]);
        }
    } 
    } else if (std::is_same_v<value_t, float>) {
    /*
    (64, 512) in terms of bytes
    COLS = 4
    ROWS = 4
    */
    // float32 implementation
    int lane_id = threadIdx.x % 32;
    int thread_row = lane_id / 8; 
    int thread_col = lane_id % 8;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            op()(&gmem[warp_id * 16 * HEAD_DIM + r * 4 * HEAD_DIM + thread_row * HEAD_DIM + c * 8 * 4 + thread_col * 4],
                 &smem[warp_id * 16 * HEAD_DIM + r * 4 * HEAD_DIM + thread_row * HEAD_DIM + c * 8 * 4 + thread_col * 4]);
        }
    }

}
}

__forceinline__ __device__ void copy_output_and_transform_(
	__half *gmem,
	float* smem,
    const int warp_id
){
    half2* gmem_h2 = reinterpret_cast<half2*>(gmem);
    float2* smem_f2 = reinterpret_cast<float2*>(smem);

    for(int i = 0; i < 128 * 32; i += blockDim.x) {
        int index = threadIdx.x + i;
        gmem_h2[index] = __float22half2_rn(smem_f2[index]);
    }
}