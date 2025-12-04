#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" void run_normal_attention(
    const __half* dQ,
    const __half* dK,
    const __half* dV,
    __half* dO,
    int B,
    int H,
    int L,
    int D,
    int causal = 0,
    cudaStream_t stream = 0,
    float* elapsed_ms = nullptr);