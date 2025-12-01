#pragma once


template <int B, int H, int L, int D, int tile, int shared_mem_size>
__global__ void flash_attention_kernel(
	const __half* __restrict__ Q,
	const __half* __restrict__ K,
	const __half* __restrict__ V,
	__half* __restrict__ O
);