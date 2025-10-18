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

__global__ void normal_attention_kernel(
	const __half* __restrict__ Q,
	const __half* __restrict__ K,
	const __half* __restrict__ V,
	__half* __restrict__ O,
	int B, int H, int L, int D,
	int causal 
) {
	int bh = blockIdx.x;
	if (bh >= B * H) return;
	int b = bh / H;
	int h = bh % H;

	const size_t LD = (size_t)L * (size_t)D;
	const __half* __restrict__ Qbh = Q + ((size_t)b * H + h) * LD;
	const __half* __restrict__ Kbh = K + ((size_t)b * H + h) * LD;
	const __half* __restrict__ Vbh = V + ((size_t)b * H + h) * LD;
	__half* __restrict__ Obh = O + ((size_t)b * H + h) * LD;

	float scale = rsqrtf((float)D);

	for (int i = threadIdx.x; i < L; i += blockDim.x) {
		const __half* __restrict__ Qi = Qbh + (size_t)i * D;

		float max_score = -1e30f;
		for (int j = 0; j < L; ++j) {
			if (causal && j > i) continue;
			const __half* __restrict__ Kj = Kbh + (size_t)j * D;
			float dot = 0.0f;
			for (int d = 0; d < D; ++d) {
				dot += __half2float(Qi[d]) * __half2float(Kj[d]);
			}
			float score = dot * scale;
			max_score = fmaxf(max_score, score);
		}

		if (max_score <= -1e29f) max_score = 0.0f;

		float denom = 0.0f;
		for (int j = 0; j < L; ++j) {
			if (causal && j > i) continue;
			const __half* __restrict__ Kj = Kbh + (size_t)j * D;
			float dot = 0.0f;
			for (int d = 0; d < D; ++d) {
				dot += __half2float(Qi[d]) * __half2float(Kj[d]);
			}
			float score = dot * scale;
			float p = __expf(score - max_score);
			denom += p;
		}
		denom = fmaxf(denom, 1e-20f);

		__half* __restrict__ Oi = Obh + (size_t)i * D;
		for (int d = 0; d < D; ++d) {
			Oi[d] = __float2half(0.0f);
		}
		for (int j = 0; j < L; ++j) {
			if (causal && j > i) continue;
			const __half* __restrict__ Kj = Kbh + (size_t)j * D;
			float dot = 0.0f;
			for (int d = 0; d < D; ++d) {
				dot += __half2float(Qi[d]) * __half2float(Kj[d]);
			}
			float score = dot * scale;
			float p = __expf(score - max_score) / denom;

			const __half* __restrict__ Vj = Vbh + (size_t)j * D;
			for (int d = 0; d < D; ++d) {
				float acc = __half2float(Oi[d]) + p * __half2float(Vj[d]);
				Oi[d] = __float2half(acc);
			}
		}
	}
}

	// Host wrapper implementation for reuse
	#include "attention_kernels.h"

	void run_normal_attention(
		const __half* dQ,
		const __half* dK,
		const __half* dV,
		__half* dO,
		int B,
		int H,
		int L,
		int D,
		int causal,
		cudaStream_t stream,
		float* elapsed_ms
	) {
		dim3 grid((unsigned)(B * H));
		dim3 block((unsigned)min(L, 128));

		cudaEvent_t start_ev = nullptr, stop_ev = nullptr;
		if (elapsed_ms) {
			CUDA_CHECK(cudaEventCreate(&start_ev));
			CUDA_CHECK(cudaEventCreate(&stop_ev));
			CUDA_CHECK(cudaEventRecord(start_ev, stream));
		}

		normal_attention_kernel<<<grid, block, 0, stream>>>(dQ, dK, dV, dO, B, H, L, D, causal);
		CUDA_CHECK(cudaGetLastError());

		if (elapsed_ms) {
			CUDA_CHECK(cudaEventRecord(stop_ev, stream));
			CUDA_CHECK(cudaEventSynchronize(stop_ev));
			CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start_ev, stop_ev));
			CUDA_CHECK(cudaEventDestroy(start_ev));
			CUDA_CHECK(cudaEventDestroy(stop_ev));
		}
	}


