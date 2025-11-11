#include <cuda_utils.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda::wmma;



/*

Tensor Flash Attention 2 CUDA Kernel
This kernel implements the Flash Attention mechanism for efficient self-attention computation.
It utilizes shared memory and warp-level primitives to optimize memory access patterns and reduce
the overall computation time.


Author: Konrad Burdach

Assumptions:
- Input tensors Q, K, V, O are in half-precision (__half).
- Softmax is computed in float32 for numerical stability.
- 4 warps per block each processing 16x16 tiles
- head_dim is 128

*/
__global__ void flash_attention_kernel(
	const __half* __restrict__ Q,
	const __half* __restrict__ K,
	const __half* __restrict__ V,
	__half* __restrict__ O,
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
    int K_offset = batch_id * H * L * D + head_id * L * D + tile_id * tile * D;
    int V_offset = batch_id * H * L * D + head_id * L * D + tile_id * tile * D;
    int O_offset = batch_id * H * L * D + head_id * L * D + tile_id * tile * D;
    int O_offset_shmem = D * tile;;
    int K_V_offset_shmem = D * tile + O_offset_shmem;
    int tile_offset_shmem = D * tile + O_offset_shmem;
    int running_max_offset_shmem =  tile * tile + tile_offset_shmem;



    int B_r = tile;
    int T_r = L / 64;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0);




    // loading Q into shared memory
    copy_block_GSM<GM2SM_async<__half>, __half>(
        ( __half*)&Q[Q_offset],
        ( __half*)&shared_mem[0],
        lane_id
    );
    cp_async_commit();


    // loading O into shared memory
    copy_block_GSM<GM2SM_async<__half>, __half>(
        ( __half*)&O[O_offset],
        ( __half*)&shared_mem[O_offset_shmem],
        lane_id
    );
    cp_async_commit();

    for(int i = 0; i < T_r; i++){
        // loading K into shared memory
        copy_block_GSM<GM2SM_async<__half>, __half>(
            ( __half*)&K[K_offset + i * tile * D],
            ( __half*)&shared_mem[K_V_offset_shmem],
            lane_id
        );
        cp_async_commit();
        cp_async_wait<0>();
        for(int j = 0; j < 64; j += 16) {
            for (int k = 0; k < 128; k += 16) {
            
                int col = k;
                int row = j;

                // Load the inputs
                nvcuda::wmma::load_matrix_sync(a_frag, &shared_mem[lane_id * 16 * D + col], 128);
                nvcuda::wmma::load_matrix_sync(b_frag, &shared_mem[row * D + col], 128);

                // Perform the matrix multiplication
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            nvcuda::wmma::store_matrix_sync(&shared_mem[tile_offset_shmem + lane_id * 16 * D + j], c_frag, 128, nvcuda::wmma::mem_row_major);
        }
}

    












    __syncthreads();









	



}


int main(){
    // test parameters
    const int B = 10;
    const int H = 8;
    const int L = 64 * 10;
    const int D = 128;



    // hyperparameters
    const int number_of_warps = 4;
    const int tile = 16 * 4;
    int shared_mem_needed = D * tile * sizeof(__half); // queries
    shared_mem_needed += D * tile * sizeof(__half); // output
    shared_mem_needed += D * tile * sizeof(__half); // keys + values idepdendently
    shared_mem_needed += tile * tile * sizeof(__half); // tile
    shared_mem_needed += 64 * 2 * sizeof(__half); // running max sum per tile






    int number_of_blocks = B * H * (L / tile);
    int threads_per_block = number_of_warps * 32;

    dim3 gridDim(L/ tile, B, H);
    flash_attention_kernel<<<gridDim, threads_per_block, shared_mem_needed>>>(Q, K, V, O, B, H, L, D, tile);

    return 0;
}