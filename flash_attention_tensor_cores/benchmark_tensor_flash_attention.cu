#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include "tensor_flash_attention.cuh"   // must declare flash_attention_kernel

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void init_half(__half* ptr, size_t n, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < n) {
        // simple deterministic pseudo-random pattern
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

    std::vector<int> seqs = {8192};
    std::vector<std::pair<int,int>> bh_sets = {
        {2, 16}
    };

    // Pre-allocate for max config
    int maxB = 0, maxH = 0, maxL = 0;
    for (auto &p : bh_sets) {
        maxB = std::max(maxB, p.first);
        maxH = std::max(maxH, p.second);
    }
    for (int L : seqs) maxL = std::max(maxL, L);

    size_t maxElems = (size_t)maxB * maxH * maxL * D;
    size_t bytes = maxElems * sizeof(__half);

    __half *dQ, *dK, *dV;
    float *dO;
    CHECK_CUDA(cudaMalloc(&dQ, bytes));
    CHECK_CUDA(cudaMalloc(&dK, bytes));
    CHECK_CUDA(cudaMalloc(&dV, bytes));
    CHECK_CUDA(cudaMalloc(&dO, maxElems * sizeof(float)));

    // Initialize once
    int threads = 256;
    int blocks = (int)((maxElems + threads - 1) / threads);
    init_half<<<blocks, threads>>>(dQ, maxElems, 1234ULL);
    init_half<<<blocks, threads>>>(dK, maxElems, 2345ULL);
    init_half<<<blocks, threads>>>(dV, maxElems, 3456ULL);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Shared memory size (must match kernel layout)
    // Total halves = 3*D*tile + tile*tile + 4*16*2
    // int shared_mem_needed = D * tile * sizeof(__half); // queries
    // shared_mem_needed += D * tile * sizeof(float); // output
    // shared_mem_needed += D * tile * sizeof(__half); // keys + values idepdendently
    // shared_mem_needed += tile * tile * sizeof(float); // tile
    // shared_mem_needed += 64 * sizeof(float); // running max
    // shared_mem_needed += 64 * sizeof(float); // running sum
    // shared_mem_needed += 64 * sizeof(float); // local max
    // shared_mem_needed += 16 * 16 * 4 * sizeof(float); 

    int shared_mem_needed = D * tile * sizeof(__half); // queries
    shared_mem_needed += D * tile * sizeof(float); // output
    shared_mem_needed += D * tile * sizeof(__half); // keys + values idepdendently
    shared_mem_needed += tile * tile * sizeof(float); // tile
    shared_mem_needed += 64 * sizeof(float); // running max
    shared_mem_needed += 64 * sizeof(float); // running sum
    shared_mem_needed += 64 * sizeof(float); // local max
    shared_mem_needed += 16 * 16 * 4 * sizeof(float); 

    printf("Benchmark Tensor Flash Attention Kernel\n");
    printf("D=%d tile=%d warmup=%d reps=%d shared_mem=%.2f KB\n",
           D, tile, warmup, reps, shared_mem_needed / 1024.0);

    for (auto &bh : bh_sets) {
        int B = bh.first;
        int H = bh.second;
        for (int L : seqs) {
            if (L % tile != 0) {
                printf("Skipping B=%d H=%d L=%d (L not multiple of tile)\n", B, H, L);
                continue;
            }

            dim3 grid(L / tile, B, H);
            dim3 block(128); // 4 warps
            
            // Increase dynamic shared memory limit for this kernel
            CHECK_CUDA(cudaFuncSetAttribute(
                flash_attention_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_needed
            ));

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

            // Rough FLOPs: per head ~ 2 * L * L * D (QK^T + softmax*V)
            // Multiply by B*H
            double flops = 4.0 * (double)L * (double)L * (double)D* (double)H;
            double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

            printf("B=%d H=%d L=%d: avg_time=%.3f ms, est_TFLOPs=%.3f\n",
                   B, H, L, avg_ms, tflops);
        }
    }

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    return 0;
}