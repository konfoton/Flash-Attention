#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include "normal_attention.cuh"
#include "wrapper.cuh"

extern "C" void run_naive_attention_host_half(
    const __half* hQ,
    const __half* hK,
    const __half* hV,
    __half* hO,
    int B,
    int H,
    int L,
    int D,
    int causal,
    cudaStream_t stream,
    float* elapsed_ms)
{
    size_t total = (size_t)B * H * L * D;
    size_t bytes = total * sizeof(__half);
    __half *dQ=nullptr,*dK=nullptr,*dV=nullptr,*dO=nullptr;
    if (cudaMalloc(&dQ, bytes) != cudaSuccess ||
        cudaMalloc(&dK, bytes) != cudaSuccess ||
        cudaMalloc(&dV, bytes) != cudaSuccess ||
        cudaMalloc(&dO, bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed in run_naive_attention_host_half");
    }
    cudaMemcpyAsync(dQ, hQ, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dK, hK, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dV, hV, bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    run_normal_attention(dQ, dK, dV, dO, B, H, L, D, causal, stream, elapsed_ms);

    cudaMemcpyAsync(hO, dO, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}
