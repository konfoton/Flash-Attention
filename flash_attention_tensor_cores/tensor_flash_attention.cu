#include "cuda_utils.cuh"
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "tensor_flash_attention.cuh"
#include <stdio.h>

using namespace nvcuda;

/*

Tensor Flash Attention 2 CUDA Kernel
This kernel implements the Flash Attention mechanism for efficient self-attention (is_casual = False) computation.
It utilizes shared memory and warp-level primitives to optimize memory access patterns and reduce
the overall computation time.


Author: Konrad Burdach

Assumptions:
- Input tensors Q, K, V, O are in half-precision (__half).
- Softmax is computed in float32 (to do) for numerical stability.
- 4 warps per block each processing 16x16 tiles
- head_dim is 128

*/


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
){
    extern __shared__ __half shared_mem[];


    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tile_id = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int tid = threadIdx.x;


    int Q_offset = batch_id * H * L * D + head_id * L * D + tile_id * tile * D;
    int K_offset = batch_id * H * L * D + head_id * L * D;
    int V_offset = batch_id * H * L * D + head_id * L * D;
    int O_offset = batch_id * H * L * D + head_id * L * D + tile_id * tile * D;
    int O_offset_shmem = D * tile;
    int K_V_offset_shmem = D * tile * 2 + O_offset_shmem;
    int tile_offset_shmem = D * tile + K_V_offset_shmem;
    int running_max_offset_shmem =  tile * tile * 2 + tile_offset_shmem;
    int local_max_offset_shmem = 64 * 2 * 2 + running_max_offset_shmem;
    int offset_to_process = 64 * 2 + local_max_offset_shmem;



    int B_r = tile;
    int T_r = L / 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);


    // loading Q into shared memory
    copy_block_GSM<GM2SM_async<__half>, __half>(
        ( __half*)&Q[Q_offset],
        ( __half*)&shared_mem[0],
        warp_id
    );
    cp_async_commit();


    // Initialize O buffer in shared memory as float zeros
    for (int idx = tid; idx < D * tile; idx += blockDim.x) {
        ((float*)&shared_mem[O_offset_shmem])[idx] = 0.0f;
    }
    // setting running max to -INF and sum to zero
    if(lane_id < 16){
           ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 16 * 2 + lane_id * 2 + 0)[0] = -1e30f; // -INF for __half
           ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 16 * 2 + lane_id * 2 + 1)[0] = 0.0f;
    }

    __syncwarp();


    for(int i = 0; i < T_r; i++){
        // loading K into shared memory
        copy_block_GSM<GM2SM_async<__half>, __half>(
            ( __half*)&K[K_offset + i * tile * D],
            ( __half*)&shared_mem[K_V_offset_shmem],
            warp_id
        );

        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();

        for(int j = 0; j < 64; j += 16) {

            for (int k = 0; k < 128; k += 16) {
            
                int col_a = k;
                int col_b = j;

                // Load the inputs
                nvcuda::wmma::load_matrix_sync(a_frag, &shared_mem[warp_id * 16 * D + col_a], 128);
                nvcuda::wmma::load_matrix_sync(b_frag_col, &shared_mem[K_V_offset_shmem + col_b * D + col_a], 128);

                // Perform the matrix multiplication
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag_col, c_frag);
            }

            nvcuda::wmma::store_matrix_sync(((float*)&shared_mem[tile_offset_shmem] + warp_id * 16 * 64 + j), c_frag, 64, nvcuda::wmma::mem_row_major);
            nvcuda::wmma::fill_fragment(c_frag, 0.0f);

        }
        // loading V into shared memory
        copy_block_GSM<GM2SM_async<__half>, __half>(
            ( __half*)&V[V_offset + i * tile * D],
            ( __half*)&shared_mem[K_V_offset_shmem],
            warp_id
        );

        cp_async_commit();
        /*
        
        */

        /*
        warp level reduction to compute softmax
        tile is size (64, 64) each warp computes (16, 64)
        */
        float max_local = 0.0f;
        float scale = 1.0f / sqrtf((float)D);
        float max_prev = 0.0f;
        float max = 0.0f;
        float sum = 0.0f;
        float prev_sum = 0.0f;
        float scale_second = 1.0f;
        for(int j = 0; j < 16; j++){
            int offset_thread = warp_id * 16 * 64 + j * 64 + lane_id;
            float val1 = (((float*)&shared_mem[tile_offset_shmem])[offset_thread]) * scale;
            float val2 = (((float*)&shared_mem[tile_offset_shmem])[offset_thread + 32]) * scale;
            max_local = fmaxf(val1, val2);

            for(int offset = 16; offset >= 1; offset = offset / 2){
                max_local = fmaxf(max_local, __shfl_down_sync(0xffffffff, max_local, offset));
            }

            max_local = __shfl_sync(0xffffffff, max_local, 0);


            if(lane_id % 32 == 0){
                ((float*)&shared_mem[local_max_offset_shmem])[warp_id * 16 + j] = max_local;
            }
            /*synchornization is done here so i can write to shared mem*/
            float exp1 = expf(val1 - max_local);
            float exp2 = expf(val2 - max_local);
            int new_offset_thread = warp_id * 16 * 128 + j * 128 + lane_id;
            shared_mem[tile_offset_shmem + new_offset_thread] = __float2half(exp1);
            shared_mem[tile_offset_shmem + new_offset_thread + 32] = __float2half(exp2);


            max_prev = ((float*)&shared_mem[running_max_offset_shmem])[warp_id * 16 * 2 + j * 2 + 0];
            max = fmaxf(max_prev, max_local);
            if(lane_id % 32 == 0){
                ((float*)&shared_mem[running_max_offset_shmem])[warp_id * 16 * 2 + j * 2 + 0] = max;
            }

            sum = exp1 + exp2;
            for(int offset = 16; offset >= 1; offset = offset / 2){
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }



            prev_sum = ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 2 * 16 + j * 2 + 1)[0];
            scale_second = expf(max_prev - max);
            for(int k = 0; k < 128; k += 32){
                int offset_thread_out = warp_id * 16 * 128 + j * 128 + lane_id + k;
                float val = (((float*)&shared_mem[O_offset_shmem]) + offset_thread_out)[0];
                (((float*)&shared_mem[O_offset_shmem]) + offset_thread_out)[0] = scale_second * val;
            }

            if(lane_id % 32 == 0){
                float new_sum = prev_sum * expf(max_prev - max) + sum * expf(max_local - max);
                ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 2 * 16 + j * 2 + 1)[0] = new_sum;

                if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp_id == 0 && j == 0) {
                     printf("i=%d j=%d max_prev=%f max_local=%f max=%f prev_sum=%f sum=%f new_sum=%f scale_second=%f scale_update=%f\n",
                        i, j, max_prev, max_local, max, prev_sum, sum, new_sum, scale_second, expf(max_local - max));
                }
            }


        }

        /*After calculating softmax i will cast the output to float16
        I will use __float22half2_rn to convert two float values to __half2
        (SASS support only double conversinon if single is used)
        round to nearest even (statiscial bias is minimized)
        */

        cp_async_wait<0>();
        __syncthreads();

        // compute output
        for (int j = 0; j < 128; j += 16) {
            nvcuda::wmma::fill_fragment(c_frag, 0.0f);
            // load_matrix_sync(c_frag, (((float*)&shared_mem[O_offset_shmem]) + warp_id * 16 * D + j), 128, nvcuda::wmma::mem_row_major);
            for(int k = 0; k < 64; k += 16) {

                int col_a = k;
                int col_b = j;

                // Load the inputs
                nvcuda::wmma::load_matrix_sync(a_frag, &shared_mem[tile_offset_shmem + warp_id * 128 * 16 + col_a], 128);
                nvcuda::wmma::load_matrix_sync(b_frag_row, &shared_mem[K_V_offset_shmem + col_a * D + col_b], 128);

                // Perform the matrix multiplication
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag_row, c_frag);
        }
            // nvcuda::wmma::store_matrix_sync((((float*)&shared_mem[O_offset_shmem]) + warp_id * 16 * D + j), c_frag, 128, nvcuda::wmma::mem_row_major);
            nvcuda::wmma::store_matrix_sync((((float*)&shared_mem[offset_to_process]) + warp_id * 16 * 16), c_frag, 16, nvcuda::wmma::mem_row_major);

            /* UPDATING to NEW MAX MOVING TO REAL O AND MUTIPLYING RESULT BY DIAG(l_{i}^new)^-1 */
            int group = lane_id / 16;
            int new_lane = lane_id % 16;
            float max_local;
            float max_global;
            float new_sum;
            for(int m = 0; m < 8; m++){
                max_local = ((float*)&shared_mem[local_max_offset_shmem])[warp_id * 16 + m * 2 + group];
                max_global = ((float*)&shared_mem[running_max_offset_shmem])[warp_id * 16 * 2 + m * 4 + group * 2 + 0];
                new_sum = ((float*)&shared_mem[running_max_offset_shmem])[warp_id * 16 * 2 + m * 4  + group * 2 + 1];


                float scale_update = expf(max_local - max_global);
                int offset_thread_within = warp_id * 16 * 16 + m * 2 * 16 + group * 16 + new_lane;
                float val = (((float*)&shared_mem[offset_to_process]) + offset_thread_within)[0];
                int offset_thread_out_final_output = warp_id * 16 * 128 + m * 2 * 128 + group * 128 + new_lane + j;
                (((float*)&shared_mem[O_offset_shmem]) + offset_thread_out_final_output)[0] += val * scale_update;
                if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp_id == 0 && m == 0 && j == 0){
                    printf("m=%d group=%d new_lane=%d max_local=%f max_global=%f new_sum=%f scale_update=%f val=%f final_output=%f\n",
                        m, group, new_lane, max_local, max_global, new_sum, scale_update, val,
                        (((float*)&shared_mem[O_offset_shmem]) + offset_thread_out_final_output)[0]);
                }
            }
        


        }
        __syncthreads();


        



        // float max = 0.0f;
        // float max_prev = 0.0f;
        // float local_max = 0.0f;
        // float sum = 0.0f;
        // for(int j = 0; j < 16; j++){
        //     local_max = ((float*)&shared_mem[local_max_offset_shmem])[warp_id * 16 + j];
            

        // }



        /* NUMERICAL STABILITY */
        // for(int j = 0; j < 16; j++){
        //     float running_sum = ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 16 * 2 + j * 2 + 1)[0];
        //     running_sum = 1.0f / running_sum;
        //     for(int k = 0; k < 128; k += 32){
        //         int offset_thread = warp_id * 16 * 128 + j * 128 + lane_id + k;
        //         ((float*)&shared_mem[O_offset_shmem])[offset_thread] = ((float*)&shared_mem[O_offset_shmem])[offset_thread] * running_sum;
        //     }
        // }

        
    }

    for(int j = 0; j < 16; j++){
        float running_sum = ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 16 * 2 + j * 2 + 1)[0];
        running_sum = 1.0f / running_sum;
        for(int k = 0; k < 128; k += 32){
            int offset_thread = warp_id * 16 * 128 + j * 128 + lane_id + k;
            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp_id == 0 && j == 1){
                printf("final output check: offset_thread=%f value=%f\n",1.0f / running_sum, ((float*)&shared_mem[O_offset_shmem])[offset_thread]);
        }
            ((float*)&shared_mem[O_offset_shmem])[offset_thread] = ((float*)&shared_mem[O_offset_shmem])[offset_thread] * running_sum;
        
    }
}


    
    // write back output to global memory (store as float)
    copy_block_GSM<SM2GM<float>, float>(
        (float*)&O[O_offset],
        (float*)&shared_mem[O_offset_shmem],
        warp_id
    );
    cp_async_commit();
    cp_async_wait<0>();
        
}