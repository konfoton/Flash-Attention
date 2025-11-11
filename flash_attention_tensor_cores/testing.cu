

#pragma once
#include <cuda_utils.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "tensor_flash_attention.cuh"



int main(){
    // test parameters
    const int B = 10;
    const int H = 8;
    const int L = 64 * 10;
    const int D = 128;

    // allocate memory
    __half *Q, *K, *V, *O;
    cudaMalloc(&Q, B * H * L * D* sizeof(__half));
    cudaMalloc(&K, B * H * L * D* sizeof(__half));
    cudaMalloc(&V, B * H * L * D* sizeof(__half));
    cudaMalloc(&O, B * H * L * D* sizeof(__half));





    // hyperparameters
    const int number_of_warps = 4;
    const int tile = 16 * 4;
    int shared_mem_needed = D * tile * sizeof(__half); // queries
    shared_mem_needed += D * tile * sizeof(__half); // output
    shared_mem_needed += D * tile * sizeof(__half); // keys + values idepdendently
    shared_mem_needed += tile * tile * sizeof(__half); // tile
    shared_mem_needed += 64 * 2 * sizeof(__half); // running max sum per tile






    int number_of_blocks = B * H * (L / tile);
    int threads_per_block = number_of_warps * 32;

    dim3 gridDim(L/ tile, B, H);
    flash_attention_kernel<<<gridDim, threads_per_block, shared_mem_needed>>>(Q, K, V, O, B, H, L, D, tile);
    return 0;
}