## Flash Attention Implementations

This project explores the progression from a straightforward attention kernel to optimized Flash Attention variants, culminating in a Tensor Core implementation. The focus is on understanding memory access patterns, numerical stability (streaming softmax), and how tiling plus hardware intrinsics transform an O(n²) memory footprint into an on‑chip streaming computation.

## Implementations
1. CPU baseline – establishes correctness and acts as a performance reference.
2. Naive GPU – simple CUDA translation; memory‑bandwidth bound and materializes the attention matrix.
3. Flash Attention (CUDA cores) – tiles Q/K/V, performs online softmax, avoids full attention matrix, reduces global memory traffic.
4. Flash Attention (Tensor Cores) – leverages mixed precision and warp‑level matrix multiply‑accumulate (MMA) operations for higher throughput.

I reimplemented core ideas from Flash Attention 2 (paper: https://arxiv.org/pdf/2307.08691)



## Benchmark Methodology (RTX 3090)
Unless noted otherwise:
* Precision: FP16 inputs with FP32 accumulation (Tensor Core path).
* Head dimension D = 128.
* Tile size = 64.
* Warmup = 3 runs, measurement repetitions = 5.



## Comparison Table
| B | H | L | D | PyTorch Time (ms) | PyTorch TFLOPs | Custom Time (ms) | Custom TFLOPs | Time Ratio (Custom / PyTorch) | TFLOPs Ratio (Custom / PyTorch) |
|---|---|----|----|------------------|----------------|------------------|--------------|-------------------------------|---------------------------------|
|32 |16 | 512  |128 | 1.677 | 40.98 | 7.307 | 9.41 | 4.36 | 0.23 |
|16 |16 |1024  |128 | 2.712 | 50.68 |14.039 | 9.79 | 5.18 | 0.19 |
| 8 |16 |2048  |128 | 4.529 | 60.69 |25.194 |10.91 | 5.56 | 0.18 |
| 4 |16 |4096  |128 | 8.527 | 64.47 |50.269 |10.94 | 5.90 | 0.17 |
| 2 |16 |8192  |128 |16.840 | 65.31 |100.400|10.95 | 5.96 | 0.17 |
| 1 |16 |16384 |128 |31.060 | 70.79 |200.591|10.96 | 6.46 | 0.15 |

Note:
* PyTorch's implementation incorporates advanced scheduling, fusion, and kernel autotuning not yet replicated here.

## Interpreting Performance
The widening gap at higher sequence lengths reflects superior memory utilization and kernel fusion strategies in the production implementation. Closing this gap would require:
* using mma intruction for explicit registers layout and swizzing to avoid memory conflicts


## Future Work
* Replace WMMA with low-level MMA fragment layout control to shrink shared memory staging.
* Add causal & padding mask support.
* Profile with Nsight Compute to quantify memory vs compute bottlenecks.
* using PyCUDA add python API
* Extend benchmarks: variable head dimension, multi‑GPU, different batch sizes.
