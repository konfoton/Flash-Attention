#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>


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

int main(int argc, char** argv) {
	int B = 8, H = 4, L = 40, D = 64;
	int causal = 0;
	if (argc >= 5) {
		B = std::atoi(argv[1]);
		H = std::atoi(argv[2]);
		L = std::atoi(argv[3]);
		D = std::atoi(argv[4]);
	}
	printf("Running attention with B=%d H=%d L=%d D=%d causal=%d\n", B, H, L, D, 0);

	size_t total_elems = (size_t)B * H * L * D;
	size_t bytes = total_elems * sizeof(float);

	std::vector<float> hQ(total_elems), hK(total_elems), hV(total_elems), hO(total_elems, 0.0f), hOref(total_elems, 0.0f);

	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
	for (size_t i = 0; i < total_elems; ++i) {
		hQ[i] = dist(rng);
		hK[i] = dist(rng);
		hV[i] = dist(rng);
	}
	
    auto t0 = std::chrono::high_resolution_clock::now();
    attention_cpu(hQ, hK, hV, hOref, B, H, L, D, causal != 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU reference time: %.3f ms (B*H=%d, L=%d, D=%d)\n", cpu_ms, B*H, L, D);
	printf("%f", hOref[1231]);

	return 0;
}