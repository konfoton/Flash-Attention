#include "tensor_flash_attention.cuh"
#include <cuda_fp16.h>

#ifndef WRAPPER_CUH
#define WRAPPER_CUH

extern "C" {
void run_tensor_flash_attention(
    const __half* dQ,
    const __half* dK,
    const __half* dV,
    __half* dO,
    int B,
    int H,
    int L,
    int D,
    int tile,
    cudaStream_t stream = 0,
    float* elapsed_ms = nullptr);

void run_tensor_flash_attention_host_half(
    const __half* hQ,
    const __half* hK,
    const __half* hV,
    __half* hO,
    int B,
    int H,
    int L,
    int D,
    int tile,
    cudaStream_t stream = 0,
    float* elapsed_ms = nullptr);
}

#endif
