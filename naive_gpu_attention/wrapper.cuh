#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {
void run_naive_attention_host_half(
    const __half* hQ,
    const __half* hK,
    const __half* hV,
    __half* hO,
    int B,
    int H,
    int L,
    int D,
    int causal = 0,
    cudaStream_t stream = 0,
    float* elapsed_ms = nullptr);
}
