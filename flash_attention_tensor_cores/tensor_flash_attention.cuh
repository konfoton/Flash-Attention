#pragma once

__global__ void flash_attention_kernel(
	const __half* __restrict__ Q,
	const __half* __restrict__ K,
	const __half* __restrict__ V,
	float* __restrict__ O,
	int B, 
    int H, 
    int L, 
    int D, 
    int tile
);