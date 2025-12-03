#include "wrapper.cuh"
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <cuda.h>
// (Float host wrapper removed per user request; inputs are provided as half already.)

// Host-facing convenience: takes half host inputs and copies to device, returns half output
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
    cudaStream_t stream,
    float* elapsed_ms){
    size_t total = static_cast<size_t>(B) * H * L * D;
    size_t bytes = total * sizeof(__half);
    __half *dQ=nullptr,*dK=nullptr,*dV=nullptr,*dO=nullptr;
    if(cudaMalloc(&dQ, bytes)!=cudaSuccess || cudaMalloc(&dK, bytes)!=cudaSuccess || cudaMalloc(&dV, bytes)!=cudaSuccess || cudaMalloc(&dO, bytes)!=cudaSuccess){
        throw std::runtime_error("cudaMalloc failed in run_tensor_flash_attention_host_half");
    }
    cudaMemcpyAsync(dQ, hQ, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dK, hK, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dV, hV, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    run_tensor_flash_attention(dQ, dK, dV, dO, B, H, L, D, tile, stream, elapsed_ms);
    cudaMemcpyAsync(hO, dO, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

// Kernel declaration (templated) comes from tensor_flash_attention.cuh

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
    cudaStream_t stream,
    float* elapsed_ms){

    // Validate inputs minimally
    if (tile != 64) {
        // Current kernel and shared memory layout are specialized for tile=64
        // You can extend this by adjusting the formula and explicit instantiations.
        throw std::runtime_error("run_tensor_flash_attention: tile must be 64");
    }

    if (L % tile != 0) {
        throw std::runtime_error("run_tensor_flash_attention: L must be divisible by tile");
    }

    // Compute dynamic shared memory size (bytes) using the provided formula
    int shared_mem_needed = D * tile * sizeof(__half); // queries
    shared_mem_needed += D * tile * sizeof(float);     // output accumulator (float)
    shared_mem_needed += D * tile * sizeof(__half);    // keys/values (half, reused)
    shared_mem_needed += tile * tile * sizeof(float);  // score tile (float)
    shared_mem_needed += 64 * sizeof(float);           // running max
    shared_mem_needed += 64 * sizeof(float);           // running sum

    // Query device limit for max dynamic shared memory and clamp if needed

    // Configure launch parameters
    const int warps = 4; // matches kernel usage
    dim3 block(warps * 32);
    dim3 grid(L / tile, B, H);

    // Optional timing
    cudaEvent_t startEvt, stopEvt;
    bool do_timing = (elapsed_ms != nullptr);
    if (do_timing) {
        cudaEventCreate(&startEvt);
        cudaEventCreate(&stopEvt);
        cudaEventRecord(startEvt, stream);
    }

    // Dispatch based on batch size (B) to the corresponding explicit instantiation
    // Supported B values: 32,16,8,4,2,1
    switch (B) {
        case 32:
            {
                cudaError_t attrErr = cudaFuncSetAttribute(
                    (const void*)flash_attention_kernel<32, 16, 512, 128, 64>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_mem_needed);
                if (attrErr != cudaSuccess) {
                    throw std::runtime_error(std::string("cudaFuncSetAttribute failed (B=32): ") + cudaGetErrorString(attrErr));
                }
            }
            flash_attention_kernel<32, 16, 512, 128, 64><<<grid, block, shared_mem_needed, stream>>>(dQ, dK, dV, dO);
            break;
        case 16:
            {
                cudaError_t attrErr = cudaFuncSetAttribute(
                    (const void*)flash_attention_kernel<16, 16, 1024, 128, 64>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_mem_needed);
                if (attrErr != cudaSuccess) {
                    throw std::runtime_error(std::string("cudaFuncSetAttribute failed (B=16): ") + cudaGetErrorString(attrErr));
                }
            }
            flash_attention_kernel<16, 16, 1024, 128, 64><<<grid, block, shared_mem_needed, stream>>>(dQ, dK, dV, dO);
            break;
        case 8:
            {
                cudaError_t attrErr = cudaFuncSetAttribute(
                    (const void*)flash_attention_kernel<8, 16, 2048, 128, 64>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_mem_needed);
                if (attrErr != cudaSuccess) {
                    throw std::runtime_error(std::string("cudaFuncSetAttribute failed (B=8): ") + cudaGetErrorString(attrErr));
                }
            }
            flash_attention_kernel<8, 16, 2048, 128, 64><<<grid, block, shared_mem_needed, stream>>>(dQ, dK, dV, dO);
            break;
        case 4:
            {
                cudaError_t attrErr = cudaFuncSetAttribute(
                    (const void*)flash_attention_kernel<4, 16, 4096, 128, 64>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_mem_needed);
                if (attrErr != cudaSuccess) {
                    throw std::runtime_error(std::string("cudaFuncSetAttribute failed (B=4): ") + cudaGetErrorString(attrErr));
                }
            }
            flash_attention_kernel<4, 16, 4096, 128, 64><<<grid, block, shared_mem_needed, stream>>>(dQ, dK, dV, dO);
            break;
        case 2:
            {
                cudaError_t attrErr = cudaFuncSetAttribute(
                    (const void*)flash_attention_kernel<2, 16, 8192, 128, 64>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_mem_needed);
                if (attrErr != cudaSuccess) {
                    throw std::runtime_error(std::string("cudaFuncSetAttribute failed (B=2): ") + cudaGetErrorString(attrErr));
                }
            }
            flash_attention_kernel<2, 16, 8192, 128, 64><<<grid, block, shared_mem_needed, stream>>>(dQ, dK, dV, dO);
            break;
        case 1:
            {
                cudaError_t attrErr = cudaFuncSetAttribute(
                    (const void*)flash_attention_kernel<1, 16, 16384, 128, 64>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_mem_needed);
                if (attrErr != cudaSuccess) {
                    throw std::runtime_error(std::string("cudaFuncSetAttribute failed (B=1): ") + cudaGetErrorString(attrErr));
                }
            }
            flash_attention_kernel<1, 16, 16384, 128, 64><<<grid, block, shared_mem_needed, stream>>>(dQ, dK, dV, dO);
            break;
        default:
            throw std::runtime_error("run_tensor_flash_attention: unsupported batch size (B). Use one of {32,16,8,4,2,1}.");
    }

    // Optionally capture timing
    if (do_timing) {
        cudaEventRecord(stopEvt, stream);
        cudaEventSynchronize(stopEvt);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, startEvt, stopEvt);
        *elapsed_ms = ms;
        cudaEventDestroy(startEvt);
        cudaEventDestroy(stopEvt);
    }

    // Surface kernel errors to the caller (can be replaced by return code if desired)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("flash_attention_kernel launch failed: ") + cudaGetErrorString(err));
    }
}