#pragma once


template <int B, int H, int L, int D, int tile>
__global__ void flash_attention_kernel(
	const __half* __restrict__ Q,
	const __half* __restrict__ K,
	const __half* __restrict__ V,
	__half* __restrict__ O
);
extern template __global__ void flash_attention_kernel<32, 16, 512, 128, 64>(const __half*, const __half*, const __half*, __half*);
extern template __global__ void flash_attention_kernel<16, 16, 1024, 128, 64>(const __half*, const __half*, const __half*, __half*);
extern template __global__ void flash_attention_kernel<8, 16, 2048, 128, 64>(const __half*, const __half*, const __half*, __half*);
extern template __global__ void flash_attention_kernel<4, 16, 4096, 128, 64>(const __half*, const __half*, const __half*, __half*);
extern template __global__ void flash_attention_kernel<2, 16, 8192, 128, 64>(const __half*, const __half*, const __half*, __half*);
extern template __global__ void flash_attention_kernel<1, 16, 16384, 128, 64>(const __half*, const __half*, const __half*, __half*);