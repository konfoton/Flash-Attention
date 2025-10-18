#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
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

#include "attention_kernels.h"

static void init_random(std::vector<__half>& v, unsigned seed=42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto &x : v) x = __float2half(dist(rng));
}

static float max_abs_diff(const std::vector<__half>& a, const std::vector<__half>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float da = __half2float(a[i]);
        float db = __half2float(b[i]);
        float diff = fabsf(da - db);
        m = std::max(m, diff);
    }
    return m;
}

int main(int argc, char** argv) {
    int D = 64;
    int causal = 0;
    std::vector<int> seqs = {128, 256, 512, 1024};
    std::vector<std::pair<int,int>> bh_sets = {{2,4}, {4,8}, {8,8}, {8,16}, {16,16}};
    int warmup_iters = 3;
    int reps = 5;

    if (argc >= 2) D = std::atoi(argv[1]);
    if (argc >= 3) causal = std::atoi(argv[2]);

    printf("Benchmark: D=%d causal=%d | warmup=%d reps=%d\n", D, causal, warmup_iters, reps);

    for (size_t bi = 0; bi < bh_sets.size(); ++bi) {
        int B = bh_sets[bi].first;
        int H = bh_sets[bi].second;
        printf("\n== B=%d H=%d ==\n", B, H);
        for (size_t li = 0; li < seqs.size(); ++li) {
            int L = seqs[li];
            if (L % 8 != 0) continue;

            size_t total = (size_t)B * H * L * D;
            size_t bytes = total * sizeof(__half);

            std::vector<__half> hQ(total), hK(total), hV(total);
            init_random(hQ, 123);
            init_random(hK, 456);
            init_random(hV, 789);

            __half *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
            CUDA_CHECK(cudaMalloc(&dQ, bytes));
            CUDA_CHECK(cudaMalloc(&dK, bytes));
            CUDA_CHECK(cudaMalloc(&dV, bytes));
            CUDA_CHECK(cudaMalloc(&dO, bytes));
            CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice));

            // Warmup
            for (int i = 0; i < warmup_iters; ++i) {
                run_normal_attention(dQ, dK, dV, dO, B, H, L, D, causal, 0, nullptr);
            }
            for (int i = 0; i < warmup_iters; ++i) {
                run_flash_attention(dQ, dK, dV, dO, B, H, L, D, 8, 4, 0, nullptr);
            }

            // Timed runs
            float t_normal_ms = 0.f, t_flash_ms = 0.f;
            for (int i = 0; i < reps; ++i) {
                float t = 0.f;
                run_normal_attention(dQ, dK, dV, dO, B, H, L, D, causal, 0, &t);
                t_normal_ms += t;
            }
            std::vector<__half> hOn(total);
            CUDA_CHECK(cudaMemcpy(hOn.data(), dO, bytes, cudaMemcpyDeviceToHost));

            for (int i = 0; i < reps; ++i) {
                float t = 0.f;
                run_flash_attention(dQ, dK, dV, dO, B, H, L, D, 8, 4, 0, &t);
                t_flash_ms += t;
            }
            std::vector<__half> hOf(total);
            CUDA_CHECK(cudaMemcpy(hOf.data(), dO, bytes, cudaMemcpyDeviceToHost));
            printf("L=%4d | normal=%.3f ms avg | flash=%.3f ms avg\n",
             L, t_normal_ms / reps, t_flash_ms / reps );

            CUDA_CHECK(cudaFree(dQ));
            CUDA_CHECK(cudaFree(dK));
            CUDA_CHECK(cudaFree(dV));
            CUDA_CHECK(cudaFree(dO));
        }
    }

    return 0;
}
