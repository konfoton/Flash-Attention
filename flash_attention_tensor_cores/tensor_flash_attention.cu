#include "cuda_utils.cuh"
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "tensor_flash_attention.cuh"

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
           ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 16 * 2 + lane_id * 2 + 0)[0] = -10000.0f; // -INF for __half
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
        float max = 0.0f;
        float max_prev = 0.0f;
        float sum = 0.0f;
        float scale = 1.0f / sqrtf((float)D);
        for(int j = 0; j < 16; j++){
            int offset_thread = warp_id * 16 * 64 + j * 64 + lane_id;
            max_prev = ((float*)(&shared_mem[running_max_offset_shmem]))[warp_id * 16 * 2 + j * 2 + 0];
            float val1 = (((float*)&shared_mem[tile_offset_shmem])[offset_thread]) * scale;
            float val2 = (((float*)&shared_mem[tile_offset_shmem])[offset_thread + 32]) * scale;
            max = fmaxf(fmaxf(val1, val2), max_prev);
            
            for(int offset = 16; offset >= 1; offset = offset / 2){
                max = fmaxf(max, __shfl_down_sync(0xffffffff, max, offset));
            }
            
            // update running max
            if(lane_id % 32 == 0){
                ((float*)&shared_mem[running_max_offset_shmem])[warp_id * 16 * 2 + j * 2 + 0] = max;
            }

            max = __shfl_sync(0xffffffff, max, 0);
            /*synchornization is done here so i can write to shared mem*/
            float exp1 = expf(val1 - max);
            float exp2 = expf(val2 - max);
            int new_offset_thread = warp_id * 16 * 128 + j * 128 + lane_id;
            shared_mem[tile_offset_shmem + new_offset_thread] = __float2half(exp1);
            shared_mem[tile_offset_shmem + new_offset_thread + 32] = __float2half(exp2);
            sum = exp1 + exp2;

            for(int offset = 16; offset >= 1; offset = offset / 2){
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            // update running sum
            float prev_sum = ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 2 * 16 + j * 2 + 1)[0];
            if(lane_id % 32 == 0){
                float new_sum = prev_sum * expf(max_prev - max) + sum;
                ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 2 * 16 + j * 2 + 1)[0] = new_sum;
            }

            
            // updated output
            float scale_second = expf(max_prev - max);
            for(int k = 0; k < 128; k += 32){
                int offset_thread_out = warp_id * 16 * 128 + j * 128 + lane_id + k;
                float val = (((float*)&shared_mem[O_offset_shmem]) + offset_thread_out)[0];
                (((float*)&shared_mem[O_offset_shmem]) + offset_thread_out)[0] = val * scale_second * prev_sum;
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
            load_matrix_sync(c_frag, (((float*)&shared_mem[O_offset_shmem]) + warp_id * 16 * D + j), 128, nvcuda::wmma::mem_row_major);
            for(int k = 0; k < 64; k += 16) {

                int col_a = k;
                int col_b = j;

                // Load the inputs
                nvcuda::wmma::load_matrix_sync(a_frag, &shared_mem[tile_offset_shmem + warp_id * 128 * 16 + col_a], 128);
                nvcuda::wmma::load_matrix_sync(b_frag_row, &shared_mem[K_V_offset_shmem + col_a * D + col_b], 128);

                // Perform the matrix multiplication
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag_row, c_frag);
        }
            nvcuda::wmma::store_matrix_sync((((float*)&shared_mem[O_offset_shmem]) + warp_id * 16 * D + j), c_frag, 128, nvcuda::wmma::mem_row_major);
        }


        
        /* NUMERICAL STABILITY */
        for(int j = 0; j < 16; j++){
            float running_sum = ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 16 * 2 + j * 2 + 1)[0];
            running_sum = 1.0f / running_sum;
            for(int k = 0; k < 128; k += 32){
                int offset_thread = warp_id * 16 * 128 + j * 128 + lane_id + k;
                ((float*)&shared_mem[O_offset_shmem])[offset_thread] = ((float*)&shared_mem[O_offset_shmem])[offset_thread] * running_sum;
            }
        }

        
    }

    // for(int j = 0; j < 16; j++){
    //     float running_sum = ((float*)&shared_mem[running_max_offset_shmem] + warp_id * 16 * 2 + j * 2 + 1)[0];
    //     running_sum = 1.0f / running_sum;
    //     for(int k = 0; k < 128; k += 32){
    //         int offset_thread = warp_id * 16 * 128 + j * 128 + lane_id + k;
    //         ((float*)&shared_mem[O_offset_shmem])[offset_thread] = ((float*)&shared_mem[O_offset_shmem])[offset_thread] * running_sum;
    //     }
    // }


    
    // write back output to global memory (store as float)
    copy_block_GSM<SM2GM<float>, float>(
        (float*)&O[O_offset],
        (float*)&shared_mem[O_offset_shmem],
        warp_id
    );
    cp_async_commit();
    cp_async_wait<0>();
        
}