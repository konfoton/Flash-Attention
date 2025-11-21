#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
void run_flash_attention(
    const __half* dQ,
    const __half* dK,
    const __half* dV,
    __half* dO,
    int B,
    int H,
    int L,
    int D,
    int KQVO_block_y = 8,
    int number_of_warps_per_block = 4,
    cudaStream_t stream = 0,
    float* elapsed_ms = nullptr);

