// #include <cuda.h>
// #include <cuda_fp16.h>
// #include "tensor_flash_attention.cuh"
// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>


// #ifndef CUDA_CHECK
// #define CUDA_CHECK(call)                                                                 \
//     do {                                                                                 \
//         cudaError_t _cuda_check_err = (call);                                            \
//         if (_cuda_check_err != cudaSuccess) {                                            \
//             fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                         \
//                     #call, __FILE__, __LINE__, cudaGetErrorString(_cuda_check_err));     \
//             std::exit(EXIT_FAILURE);                                                     \
//         }                                                                                \
//     } while (0)
// #endif

// #ifndef CUDA_CHECK_LAST_KERNEL
// #define CUDA_CHECK_LAST_KERNEL(msg)                                                      \
//     do {                                                                                 \
//         cudaError_t _e1 = cudaGetLastError();                                            \
//         if (_e1 != cudaSuccess) {                                                        \
//             fprintf(stderr, "Kernel error after %s at %s:%d -> %s\n",                  \
//                     msg, __FILE__, __LINE__, cudaGetErrorString(_e1));                   \
//             std::exit(EXIT_FAILURE);                                                     \
//         }                                                                                \
//         CUDA_CHECK(cudaDeviceSynchronize());                                             \
//     } while (0)
// #endif



// // CPU implementation of attention for correctness testing
// #include <vector>
// #include <cmath>
// #include <algorithm>
// #include <random>
// static void attention_cpu(
//     const std::vector<__half>& Q,
//     const std::vector<__half>& K,
//     const std::vector<__half>& V,
//     std::vector<float>& O,
//     int B, int H, int L, int D,
//     bool causal
// ) {
//     auto idx = [H, L, D](int b, int h, int i, int d) -> size_t {
//         return (((size_t)b * H + h) * L + i) * D + d;
//     };
//     float scale = 1.0f / std::sqrt((float)D);
//     int tile_size = 64; // Must match the kernel's tile size

//     for (int b = 0; b < B; ++b) {
//         for (int h = 0; h < H; ++h) {
//             for (int i = 0; i < L; ++i) {
                
//                 // Per-thread state for online softmax
//                 float max_val = -1e30f;
//                 float sum_val = 0.0f;
//                 std::vector<float> acc(D, 0.0f);

//                 // Iterate over tiles to match GPU execution pattern
//                 for (int t = 0; t < L; t += tile_size) {
//                     int t_end = std::min(t + tile_size, L);
                    
//                     // 1. Compute scores for this tile
//                     // GPU: Computes scores in float, finds local max
//                     float local_max = -1e30f;
//                     std::vector<float> tile_scores(tile_size);
                    
//                     for (int j = t; j < t_end; ++j) {
//                         if (causal && j > i) {
//                             tile_scores[j - t] = -1e30f;
//                             continue;
//                         }
//                         float dot = 0.0f;
//                         for (int d = 0; d < D; ++d) {
//                             // GPU: wmma uses half inputs, accumulates in float
//                             float q_val = __half2float(Q[idx(b, h, i, d)]);
//                             float k_val = __half2float(K[idx(b, h, j, d)]);
//                             dot += q_val * k_val;
//                         }
//                         tile_scores[j - t] = dot * scale;
//                         local_max = std::max(local_max, tile_scores[j - t]);
//                     }

//                     // 2. Update global max and rescale accumulator
//                     // GPU: Updates running max, rescales O buffer
//                     float old_max = max_val;
//                     max_val = std::max(max_val, local_max);
                    
//                     float rescale_factor = std::exp(old_max - max_val);
//                     for(int d=0; d<D; ++d) acc[d] *= rescale_factor;
//                     sum_val *= rescale_factor;

//                     // 3. Compute P matrix for this tile (Unnormalized Exponentials)
//                     // GPU: exp(score - new_max), cast to HALF, stored in shared mem
//                     std::vector<__half> P_tile(tile_size);
//                     float tile_sum = 0.0f;
                    
//                     for(int j=t; j<t_end; ++j) {
//                         float s = tile_scores[j - t];
//                         // Note: GPU uses the UPDATED max_val for the current tile's exps
//                         float ex = std::exp(s - max_val);
//                         P_tile[j - t] = __float2half(ex); // Quantization happens here!
//                         tile_sum += ex; // GPU accumulates sum in float
//                     }
//                     sum_val += tile_sum;

//                     // 4. Accumulate O = O + P_tile * V_tile
//                     // GPU: wmma uses half P_tile and half V, accumulates in float
//                     for(int d=0; d<D; ++d) {
//                         float dot_pv = 0.0f;
//                         for(int j=t; j<t_end; ++j) {
//                             float p_val = __half2float(P_tile[j - t]);
//                             float v_val = __half2float(V[idx(b, h, j, d)]);
//                             dot_pv += p_val * v_val;
//                         }
//                         acc[d] += dot_pv;
//                     }
//                 }

//                 // 5. Final Normalization
//                 for (int d = 0; d < D; ++d) {
//                     O[idx(b, h, i, d)] = acc[d] / sum_val;
//                 }
//             }
//         }
//     }
// }

// int main(){
//     // test parameters
//     const int B = 1;
//     const int H = 1;
//     const int L =  64;
//     const int D = 128;



    
//     // allocate memory
//     __half *Q, *K, *V, *O;
//     CUDA_CHECK(cudaMalloc(&Q, B * H * L * D* sizeof(__half)));
//     CUDA_CHECK(cudaMalloc(&K, B * H * L * D* sizeof(__half)));
//     CUDA_CHECK(cudaMalloc(&V, B * H * L * D* sizeof(__half)));
//     CUDA_CHECK(cudaMalloc(&O, B * H * L * D* sizeof(__half)));


//     // host memory for correctness testing (compute in float32)
//     std::vector<float> Q_cpu(B * H * L * D);
//     std::vector<float> K_cpu(B * H * L * D);
//     std::vector<float> V_cpu(B * H * L * D);
//     std::vector<float> O_cpu(B * H * L * D, 0.0f);

//     // generate random float data in [0,1)
//     std::mt19937 gen(42);
//     std::normal_distribution<float> d(0.0f, 1.0f);
//     float bias = 0.5f;
//     for(size_t i = 0; i < Q_cpu.size(); i++){
//         Q_cpu[i] = d(gen) + bias;
//         K_cpu[i] = d(gen) + bias;
//         V_cpu[i] = d(gen) + bias;
//     }

//     // create temporary half buffers to copy to device
//     std::vector<__half> Q_half(Q_cpu.size());
//     std::vector<__half> K_half(K_cpu.size());
//     std::vector<__half> V_half(V_cpu.size());
//     for(size_t i = 0; i < Q_cpu.size(); ++i){
//         Q_half[i] = __float2half(Q_cpu[i]);
//         K_half[i] = __float2half(K_cpu[i]);
//         V_half[i] = __float2half(V_cpu[i]);
//     }

//     cudaMemcpy(Q, Q_half.data(), Q_half.size() * sizeof(__half), cudaMemcpyHostToDevice);
//     cudaMemcpy(K, K_half.data(), K_half.size() * sizeof(__half), cudaMemcpyHostToDevice);
//     cudaMemcpy(V, V_half.data(), V_half.size() * sizeof(__half), cudaMemcpyHostToDevice); 


//     // hyperparameters
//     const int number_of_warps = 4;
//     const int tile = 16 * 4;
//     int shared_mem_needed = D * tile * sizeof(__half); // queries
//     shared_mem_needed += D * tile * sizeof(float); // output (stored as float internally, converted to half on write)
//     shared_mem_needed += D * tile * sizeof(__half); // keys + values idepdendently
//     shared_mem_needed += tile * tile * sizeof(float); // tile
//     shared_mem_needed += 64 * sizeof(float); // running max
//     shared_mem_needed += 64 * sizeof(float); // running sum
    






//     int number_of_blocks = B * H * (L / tile);
//     int threads_per_block = number_of_warps * 32;

//     dim3 grid(L/ tile, B, H);

//     // Query device properties to diagnose kernel launch failure
//     int dev = 0;
//     cudaDeviceProp prop;
//     CUDA_CHECK(cudaGetDevice(&dev));
//     CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    
//     printf("=== Launch Configuration ===\n");
//     printf("Device: %s\n", prop.name);
//     printf("Device limits:\n");
//     printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
//     printf("  sharedMemPerBlock: %zu bytes\n", (size_t)prop.sharedMemPerBlock);
//     printf("  sharedMemPerMultiprocessor: %zu bytes\n", (size_t)prop.sharedMemPerMultiprocessor);
//     printf("\nRequested:\n");
//     printf("  Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
//     printf("  Threads per block: %d\n", threads_per_block);
//     printf("  Dynamic shared mem: %d bytes\n", shared_mem_needed);
//     printf("\nValidation:\n");
//     printf("  Threads per block OK? %s (limit: %d)\n", 
//            threads_per_block <= prop.maxThreadsPerBlock ? "YES" : "NO", prop.maxThreadsPerBlock);
//     printf("  Shared mem OK? %s (limit: %zu)\n", 
//            shared_mem_needed <= (int)prop.sharedMemPerBlock ? "YES" : "NO", (size_t)prop.sharedMemPerBlock);
//     printf("=============================\n\n");

//     // Increase dynamic shared memory limit for this kernel (RTX 3090 supports up to ~100KB)
//     CUDA_CHECK(cudaFuncSetAttribute(
//         flash_attention_kernel,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         shared_mem_needed
//     ));
//     printf("Set kernel max dynamic shared memory to: %d bytes\n\n", shared_mem_needed);

//     flash_attention_kernel<<<grid, threads_per_block, shared_mem_needed>>>(Q, K, V, O, B, H, L, D, tile);
//     CUDA_CHECK_LAST_KERNEL("flash_attention_kernel");


//     // copy back results (device stores __half)
//     std::vector<__half> O_gpu_half(B * H * L * D);
//     CUDA_CHECK(cudaMemcpy(O_gpu_half.data(), O, O_gpu_half.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    
//     // Convert to float for comparison
//     std::vector<float> O_gpu(B * H * L * D);
//     for(size_t i = 0; i < O_gpu_half.size(); ++i){
//         O_gpu[i] = __half2float(O_gpu_half[i]);
//     }

//     // compute reference results on CPU (float)
//     attention_cpu(
//         Q_half,
//         K_half,
//         V_half,
//         O_cpu,
//         B, H, L, D, false
//     );

//     // O is already float on device; no conversion needed

//     // verify correctness
//     float max_abs_error = 0.0f;
//     float mean_abs_error = 0.0f;
//     float max_rel_error = 0.0f;
//     float mean_rel_error = 0.0f;
//     double sum_sq_diff = 0.0;
//     double sum_sq_ref = 0.0;

//     for(size_t i = 0; i < O_cpu.size(); i++){
//         float diff = std::abs(O_cpu[i] - O_gpu[i]);
//         if(diff > max_abs_error) max_abs_error = diff;
//         mean_abs_error += diff;

//         float ref_abs = std::abs(O_cpu[i]);
//         float rel = diff / (ref_abs + 1e-6f);
//         if(rel > max_rel_error) max_rel_error = rel;
//         mean_rel_error += rel;

//         sum_sq_diff += (double)diff * diff;
//         sum_sq_ref += (double)O_cpu[i] * O_cpu[i];
//     }
//     mean_abs_error /= O_cpu.size();
//     mean_rel_error /= O_cpu.size();
//     float global_rel_error = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-6f);

//     printf("Mean Absolute Error: %f\n", mean_abs_error);
//     printf("Max Absolute Error: %f\n", max_abs_error);
//     printf("Mean Relative Error: %f\n", mean_rel_error);
//     printf("Max Relative Error: %f\n", max_rel_error);
//     printf("Global Relative Error: %f\n", global_rel_error);

//     for(int i = 0; i < 10; i++){
//         printf("O_cpu[%d] = %f, O_gpu[%d] = %f\n", i, O_cpu[i], i, O_gpu[i]);
//     }
//     return 0;
// }