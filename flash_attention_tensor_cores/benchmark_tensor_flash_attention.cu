#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include "tensor_flash_attention.cuh"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                                 \
    do {                                                                                 \
        cudaError_t _cuda_check_err = (call);                                            \
        if (_cuda_check_err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_cuda_check_err));     \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)
#endif

#ifndef CUDA_CHECK_LAST_KERNEL
#define CUDA_CHECK_LAST_KERNEL(msg)                                                      \
    do {                                                                                 \
        cudaError_t _e1 = cudaGetLastError();                                            \
        if (_e1 != cudaSuccess) {                                                        \
            fprintf(stderr, "Kernel error after %s at %s:%d -> %s\n",                  \
                    msg, __FILE__, __LINE__, cudaGetErrorString(_e1));                   \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
        CHECK_CUDA(cudaDeviceSynchronize());                                             \
    } while (0)
#endif

__global__ void init_half(__half* ptr, size_t n, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        float v = fmodf((float)((idx * 16807ULL + seed) & 0xFFFF), 1024.f) / 1024.f - 0.5f;
        ptr[idx] = __float2half(v);
        idx += gridDim.x * blockDim.x;
    }
}

int main() {
    const int D = 128;
    const int tile = 64; // must divide L
    const int warmup = 3;
    const int reps = 5;

    int number_of_heads = 16;
    std::vector<int> seqs = {512, 1024, 2048, 4096, 8192, 16384};
    std::vector<int> batch = {32, 16, 8, 4, 2, 1};

    // Pre-allocate for max config
    int maxB = 0, maxH = number_of_heads, maxL = 0;
    for (int b : batch) maxB = std::max(maxB, b);
    for (int l : seqs) maxL = std::max(maxL, l);

    size_t maxElems = (size_t)maxB * maxH * maxL * D;
    size_t bytes = maxElems * sizeof(__half);

    __half *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, bytes));
    CHECK_CUDA(cudaMalloc(&dK, bytes));
    CHECK_CUDA(cudaMalloc(&dV, bytes));
    CHECK_CUDA(cudaMalloc(&dO, bytes));

    // Initialize once
    int threads = 256;
    int blocks = (int)((maxElems + threads - 1) / threads);
    init_half<<<blocks, threads>>>(dQ, maxElems, 1234ULL);
    init_half<<<blocks, threads>>>(dK, maxElems, 2345ULL);
    init_half<<<blocks, threads>>>(dV, maxElems, 3456ULL);
    CHECK_CUDA(cudaDeviceSynchronize());

    int shared_mem_needed = D * tile * sizeof(__half); // queries
    shared_mem_needed += D * tile * sizeof(float); // output (stored as float internally, converted to half on write)
    shared_mem_needed += D * tile * sizeof(__half); // keys + values idepdendently
    shared_mem_needed += tile * tile * sizeof(float); // tile
    shared_mem_needed += 64 * sizeof(float); // running max
    shared_mem_needed += 64 * sizeof(float); // running sum


    CHECK_CUDA(cudaFuncSetAttribute(
        flash_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_needed
    ));
    printf("Benchmark Tensor Flash Attention Kernel\n");
    printf("D=%d tile=%d warmup=%d reps=%d shared_mem=%.2f KB\n",
           D, tile, warmup, reps, shared_mem_needed / 1024.0);




    for(int i = 0; i < seqs.size(); ++i) {
        int B = batch[i];
        int H = number_of_heads;
        int L = seqs[i];
        
        if (L % tile != 0) {
            printf("Skipping B=%d H=%d L=%d (L not multiple of tile)\n", B, H, L);
            continue;
        }

        dim3 grid(L / tile, B, H);
            dim3 block(128); // 4 warps
            // Warmup
            for (int i = 0; i < warmup; ++i) {
                flash_attention_kernel<<<grid, block, shared_mem_needed>>>(
                    dQ, dK, dV, dO, B, H, L, D, tile
                );
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            float total_ms = 0.0f;
            for (int r = 0; r < reps; ++r) {
                CHECK_CUDA(cudaEventRecord(start));
                flash_attention_kernel<<<grid, block, shared_mem_needed>>>(
                    dQ, dK, dV, dO, B, H, L, D, tile
                );
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));
                float ms;
                CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
                total_ms += ms;
            }
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));

            float avg_ms = total_ms / reps;

            // FLOPS
            double flops = 4.0 * (double)L * (double)L * (double)D* (double)H * (double)B;
            double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

            printf("B=%d H=%d L=%d: avg_time=%.3f ms, est_TFLOPs=%.3f\n",
                   B, H, L, avg_ms, tflops);
    }

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    return 0;
}