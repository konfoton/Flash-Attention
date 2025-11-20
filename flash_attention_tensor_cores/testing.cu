#include <cuda.h>
#include <cuda_fp16.h>
#include "tensor_flash_attention.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>


#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                 \
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
        CUDA_CHECK(cudaDeviceSynchronize());                                             \
    } while (0)
#endif



// CPU implementation of attention for correctness testing
#include <vector>
#include <cmath>
#include <algorithm>
static void attention_cpu(
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    std::vector<float>& O,
    int B, int H, int L, int D,
    bool causal
) {
    auto idx = [H, L, D](int b, int h, int i, int d) -> size_t {
        return (((size_t)b * H + h) * L + i) * D + d;
    };
    float scale = 1.0f / std::sqrt((float)D);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < L; ++i) {
                std::vector<float> scores(L, -1e30f);
                for (int j = 0; j < L; ++j) {
                    if (causal && j > i) continue;
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += Q[idx(b, h, i, d)] * K[idx(b, h, j, d)];
                    }
                    scores[j] = dot * scale;
                }
                float m = -1e30f;
                for (int j = 0; j < L; ++j) m = std::max(m, scores[j]);
                float denom = 0.0f;
                for (int j = 0; j < L; ++j) denom += std::exp(scores[j] - m);
                if (denom <= 0) denom = 1.0f;
                for (int d = 0; d < D; ++d) {
                    float outd = 0.0f;
                    for (int j = 0; j < L; ++j) {
                        float p = std::exp(scores[j] - m) / denom;
                        outd += p * V[idx(b, h, j, d)];
                    }
                    O[idx(b, h, i, d)] = outd;
                }
            }
        }
    }
}

int main(){
    // test parameters
    const int B = 1;
    const int H = 128;
    const int L = 4 * 64;
    const int D = 128;



    
    // allocate memory
    __half *Q, *K, *V;
    float *O;
    CUDA_CHECK(cudaMalloc(&Q, B * H * L * D* sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&K, B * H * L * D* sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&V, B * H * L * D* sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&O, B * H * L * D* sizeof(float)));


    // host memory for correctness testing (compute in float32)
    std::vector<float> Q_cpu(B * H * L * D);
    std::vector<float> K_cpu(B * H * L * D);
    std::vector<float> V_cpu(B * H * L * D);
    std::vector<float> O_cpu(B * H * L * D, 0.0f);

    // generate random float data in [0,1)
    for(size_t i = 0; i < Q_cpu.size(); i++){
        float val = __half2float(__float2half(static_cast<float>(rand()) / static_cast<float>(RAND_MAX)- 0.5f) );
        Q_cpu[i] = val;
        K_cpu[i] = val;
        V_cpu[i] = val;
    }

    // create temporary half buffers to copy to device
    std::vector<__half> Q_half(Q_cpu.size());
    std::vector<__half> K_half(K_cpu.size());
    std::vector<__half> V_half(V_cpu.size());
    for(size_t i = 0; i < Q_cpu.size(); ++i){
        Q_half[i] = __float2half(Q_cpu[i]);
        K_half[i] = __float2half(K_cpu[i]);
        V_half[i] = __float2half(V_cpu[i]);
    }

    cudaMemcpy(Q, Q_half.data(), Q_half.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(K, K_half.data(), K_half.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(V, V_half.data(), V_half.size() * sizeof(__half), cudaMemcpyHostToDevice); 


    // hyperparameters
    const int number_of_warps = 4;
    const int tile = 16 * 4;
    int shared_mem_needed = D * tile * sizeof(__half); // queries
    shared_mem_needed += D * tile * sizeof(float); // output
    shared_mem_needed += D * tile * sizeof(__half); // keys + values idepdendently
    shared_mem_needed += tile * tile * sizeof(float); // tile
    shared_mem_needed += 64 * sizeof(float); // running max
    shared_mem_needed += 64 * sizeof(float); // running sum
    shared_mem_needed += 64 * sizeof(float); // local max
    shared_mem_needed += 16 * 16 * 4 * sizeof(float); 
    






    int number_of_blocks = B * H * (L / tile);
    int threads_per_block = number_of_warps * 32;

    dim3 grid(L/ tile, B, H);

    // Query device properties to diagnose kernel launch failure
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    
    printf("=== Launch Configuration ===\n");
    printf("Device: %s\n", prop.name);
    printf("Device limits:\n");
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  sharedMemPerBlock: %zu bytes\n", (size_t)prop.sharedMemPerBlock);
    printf("  sharedMemPerMultiprocessor: %zu bytes\n", (size_t)prop.sharedMemPerMultiprocessor);
    printf("\nRequested:\n");
    printf("  Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Dynamic shared mem: %d bytes\n", shared_mem_needed);
    printf("\nValidation:\n");
    printf("  Threads per block OK? %s (limit: %d)\n", 
           threads_per_block <= prop.maxThreadsPerBlock ? "YES" : "NO", prop.maxThreadsPerBlock);
    printf("  Shared mem OK? %s (limit: %zu)\n", 
           shared_mem_needed <= (int)prop.sharedMemPerBlock ? "YES" : "NO", (size_t)prop.sharedMemPerBlock);
    printf("=============================\n\n");

    // Increase dynamic shared memory limit for this kernel (RTX 3090 supports up to ~100KB)
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_needed
    ));
    printf("Set kernel max dynamic shared memory to: %d bytes\n\n", shared_mem_needed);

    flash_attention_kernel<<<grid, threads_per_block, shared_mem_needed>>>(Q, K, V, O, B, H, L, D, tile);
    CUDA_CHECK_LAST_KERNEL("flash_attention_kernel");


    // copy back results (device stores float)
    std::vector<float> O_gpu(B * H * L * D);
    CUDA_CHECK(cudaMemcpy(O_gpu.data(), O, O_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // compute reference results on CPU (float)
    attention_cpu(
        Q_cpu,
        K_cpu,
        V_cpu,
        O_cpu,
        B, H, L, D, false
    );

    // O is already float on device; no conversion needed

    // verify correctness
    float max_error = 0.0f;
    float mean_error = 0.0f;
    for(size_t i = 0; i < O_cpu.size(); i++){
        float diff = std::abs(O_cpu[i] - O_gpu[i]);
        if(diff > max_error){
            max_error = diff;
        }
        mean_error += diff;
    }
    mean_error /= O_cpu.size();
    printf("Mean error: %f\n", mean_error);
    printf("Max error: %f\n", max_error);
    for(int i = 0; i < 10; i++){
        printf("O_cpu[%d] = %f, O_gpu[%d] = %f\n", i, O_cpu[i], i, O_gpu[i]);
    }

    // Dump data to binary files for PyTorch comparison
    auto dump_file = [](const std::string& filename, const std::vector<float>& data) {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
            file.close();
            std::cout << "Saved " << filename << std::endl;
        } else {
            std::cerr << "Unable to open file " << filename << std::endl;
        }
    };

    dump_file("Q.bin", Q_cpu);
    dump_file("K.bin", K_cpu);
    dump_file("V.bin", V_cpu);
    dump_file("O_gpu.bin", O_gpu);
    dump_file("O_cpu.bin", O_cpu);

    return 0;
}