#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <cuda_fp16.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                                                \
	do {                                                                                                \
		cudaError_t _err = (expr);                                                                      \
		if (_err != cudaSuccess) {                                                                      \
			fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__, __LINE__,              \
					cudaGetErrorString(_err));                                                          \
			std::exit(1);                                                                               \
		}                                                                                               \
	} while (0)
#endif


__global__ void flash_attention_kernel(
	const __half* __restrict__ Q,
	const __half* __restrict__ K,
	const __half* __restrict__ V,
	__half* __restrict__ O,
	int B, int H, int L, int D,
	int KQVO_block_y, int number_of_warps_per_block
){
    int batch_index = blockIdx.x; 
	int head_index = blockIdx.y; 
	int within_head = blockIdx.z;
	int warp_index = threadIdx.x / 32;
	int within_warp = threadIdx.x % 32;

	extern __shared__ float shmm[]; // store Q,K,V,O and running stats in float for numeric stability
	int warp_execution_iter = KQVO_block_y / number_of_warps_per_block;

	// loading queries
	for(int i = 0; i < warp_execution_iter; i++){
		for(int j = 0; j < D / 32; j++){
			shmm[(warp_index * warp_execution_iter + i) * D + within_warp + j * 32] = __half2float(Q[batch_index * H * L * D + head_index * L * D + within_head * KQVO_block_y * D + (warp_index * warp_execution_iter + i) * D + within_warp + j * 32]);
		}
	}
	for(int i = 0; i < warp_execution_iter; i++){
        for(int j = 0; j < D / 32; j++){
			shmm[3 * KQVO_block_y * D + (warp_index * warp_execution_iter + i) * D + within_warp + j * 32] = 0.0f;
        }
    }
	if(within_warp == 0){
	shmm[4 * KQVO_block_y * D + warp_index * 4  + 0] = -INFINITY;
	shmm[4 * KQVO_block_y * D + warp_index * 4  + 1] = 0.0f;
	shmm[4 * KQVO_block_y * D + warp_index * 4  + 2] = -INFINITY;
	shmm[4 * KQVO_block_y * D + warp_index * 4  + 3] = 0.0f;
		}
	__syncthreads();
	
	for(int m = 0; m < L / KQVO_block_y; m++){
		// loading keys and values
		for(int i = 0; i < warp_execution_iter; i++){
			for(int j = 0; j < D / 32; j++){
				shmm[1 * KQVO_block_y * D + (warp_index * warp_execution_iter + i) * D + within_warp + j * 32] = __half2float(K[batch_index * H * L * D + head_index * L * D + (warp_index * warp_execution_iter + i) * D + m * KQVO_block_y * D + within_warp + j * 32]);
				shmm[2 * KQVO_block_y * D + (warp_index * warp_execution_iter + i) * D + within_warp + j * 32] = __half2float(V[batch_index * H * L * D + head_index * L * D + (warp_index * warp_execution_iter + i) * D + m * KQVO_block_y * D + within_warp + j * 32]);
			}
		}
		__syncthreads();
		// processing keys and values
		for(int k = 0; k < warp_execution_iter; k++){
			for(int i = 0; i < KQVO_block_y; i++){
				float val = 0.0f;
				for(int j = 0; j < D / 32; j++){
					val += shmm[0 * KQVO_block_y * D + (warp_index * warp_execution_iter + k) * D + within_warp + j * 32] * shmm[1 * KQVO_block_y * D + i * D + within_warp + j * 32];
				}
				for(int offset = 16; offset > 0; offset /= 2){
						val += __shfl_down_sync(0xFFFFFFFF, val, offset);
					}
				// scale by 1/sqrt(D) to match standard attention definition
				val *= rsqrtf((float)D);
				if(within_warp == 0){
					shmm[4 * 1 * KQVO_block_y * D + 2 * KQVO_block_y + warp_index * warp_execution_iter * KQVO_block_y + k * KQVO_block_y + i] = val;
				}
			}
		}
		// assert that resulting block is XxX where X <= 32
		float sum = 0.0f;
		float max = 0.0f;
		float val = 0.0f;
		float max_prev = 0.0f;
		float elem_new;
		unsigned int mask = __ballot_sync(0xFFFFFFFF, within_warp < KQVO_block_y);
		for(int l = 0; l < warp_execution_iter; l++){
			if(within_warp < KQVO_block_y){
				max_prev = shmm[4 * 1 * KQVO_block_y * D + warp_index * 4 + l * 2];
				val = shmm[4 * 1 * KQVO_block_y * D + 2 * KQVO_block_y + warp_index * 2 * KQVO_block_y + l * KQVO_block_y + within_warp];
				max = val;
				for(int offset = 4; offset > 0; offset /= 2){
						max = fmaxf(max, __shfl_down_sync(mask, max, offset));
				}
			max = __shfl_sync(mask, max, 0);
			max = fmaxf(max_prev, max);
			elem_new = expf(val - max);
			// solving running softmax
			sum = elem_new;
			for(int offset = 4; offset > 0; offset /= 2){
					sum += __shfl_down_sync(mask, sum, offset);
				}

			// dot product with V and write to O 
			for(int i = 0; i < D; i++){

				float vall = elem_new * shmm[2 * 1 * KQVO_block_y * D + i + within_warp * D];
				for(int offset = 4; offset > 0; offset /= 2){
						vall += __shfl_down_sync(mask, vall, offset);
				}
				if(within_warp == 0){
					shmm[3 * 1 * KQVO_block_y * D + (warp_index * warp_execution_iter + l) * D + i] = shmm[3 * 1 * KQVO_block_y * D + (warp_index * warp_execution_iter + l) * D + i] * expf(max_prev - max) + vall;
				}
			}
			// updating running max and running softmax
			if(within_warp == 0){
				shmm[4 * 1 * KQVO_block_y * D + warp_index * 4 + l * 2] = max;
				shmm[4 * 1 * KQVO_block_y * D + warp_index * 4 + l * 2  + 1] =  expf(max_prev - max) * shmm[4 * 1 * KQVO_block_y * D + warp_index * 4 + l * 2  + 1] + sum;
			}
		}
		__syncwarp();
		}
		}
		__syncthreads();
		// dividing elements with running softmax
		float running_softmax = 0; 
		for(int l = 0; l < warp_execution_iter; l++){
			running_softmax = shmm[4 * 1 * KQVO_block_y * D + warp_index * 4 + l * 2  + 1];
			for(int i = 0; i < D / 32; i++){
				float outv = shmm[3 * 1 * KQVO_block_y * D + warp_index * warp_execution_iter * D + l * D + i * 32 + within_warp] / running_softmax;
				O[batch_index * H * L * D + head_index * L * D + within_head * KQVO_block_y * D + l * D + warp_index * warp_execution_iter * D + i * 32 + within_warp] = __float2half(outv);
			}
		}


	}
	

