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

    /*
    loading Q into shared memory (64, 128)
    cache line is 128 bytes
    8 threads per warp load 16 floats (128 bytes)
    therefore each warp handle (16, 128) in 8 loads (4, 64)
    */
    copy_block_GSM<GM2SM_async<__half>, __half>(
        ( __half*)&Q[Q_offset],
        ( __half*)&shared_mem[0],
        lane_id
    );
    cp_async_commit();
    cp_async_wait<0>();









    __syncthreads();






    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;



	



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
    const int shared_mem_needed


    int number_of_blocks = B * H * (L / tile);
    int threads_per_block = number_of_warps * 32;

    dim3 gridDim(L/ tile, B, H);
    flash_attention_kernel<<<gridDim, threads_per_block>>>(Q, K, V, O, B, H, L, D, tile);

    return 0;
}