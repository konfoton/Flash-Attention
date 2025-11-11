# Flash Attention (CUDA) vs Normal Attention

This project implements and benchmarks three GPU kernels for scaled dot-product attention:
- Normal (naive) attention: straightforward
- Flash-style attention: tiled, blockwise softmax that reduces memory traffic (CUDA CORES)
- Flash-style attention: tiled, blockwise softmax that reduces memory traffic (TENSOR CORES)

Both GPU paths use half-precision (__half) inputs/outputs with float accumulation for stability. A PyTorch notebook is included to compare against PyTorch’s scaled_dot_product_attention backends.

## Project structure
- `flash_attention.cu` — Flash-style kernel + host wrapper (half in/out, float math internally)
- `normal_attention.cu` — Naive kernel + host wrapper (half in/out, float math internally)
- `attention_kernels.h` — Public host APIs to launch the kernels
- `benchmark.cu` — C++ benchmark that sweeps (B, H, L) and times kernels
- `pytorch_implemtation.ipynb` — PyTorch SDPA timing on the same shapes
- `normal_attention_cpu.cpp` - cpu implementation for validity check
- `Makefile` — Build rules
- `flash_attention_tensor_cores` - currently in progress

## Build
Set the proper compute capability in the Makefile (e.g., `sm_86` for Ampere). Then build the benchmark:

```bash
make -j bench
```

Clean and rebuild:

```bash
make clean && make -j bench
```

## Run

```bash
./bench
```

Optionally open the PyTorch notebook and run the cells:
- `pytorch_implemtation.ipynb`

## Results (milliseconds)

Custom CUDA kernels in this repo (half inputs/outputs, float accumulation):

- B=2, H=4
  - L=128 | normal=4.322 | flash=0.378
  - L=256 | normal=16.758 | flash=0.844
  - L=512 | normal=60.780 | flash=2.208
  - L=1024 | normal=240.695 | flash=8.555

- B=4, H=8
  - L=128 | normal=3.829 | flash=0.561
  - L=256 | normal=15.250 | flash=2.136
  - L=512 | normal=60.741 | flash=7.786
  - L=1024 | normal=243.250 | flash=30.745

- B=8, H=8
  - L=128 | normal=4.220 | flash=1.078
  - L=256 | normal=16.846 | flash=3.903
  - L=512 | normal=66.833 | flash=15.375
  - L=1024 | normal=270.465 | flash=61.245

- B=8, H=16
  - L=128 | normal=8.013 | flash=1.973
  - L=256 | normal=31.507 | flash=7.754
  - L=512 | normal=125.664 | flash=30.636
  - L=1024 | normal=501.437 | flash=121.674

- B=16, H=16
  - L=128 | normal=15.020 | flash=3.885
  - L=256 | normal=59.481 | flash=15.320
  - L=512 | normal=236.309 | flash=60.858
  - L=1024 | normal=945.198 | flash=244.133

PyTorch SDPA (FlashAttention backend) vs math backend on the same shapes (GPU, half precision):

- B=2, H=4
  - L=128 | flash=0.0278 | math=0.1608
  - L=256 | flash=0.0268 | math=0.1603
  - L=512 | flash=0.0326 | math=0.2091
  - L=1024 | flash=0.0551 | math=0.5546

- B=4, H=8
  - L=128 | flash=0.0268 | math=0.1591
  - L=256 | flash=0.0262 | math=0.1954
  - L=512 | flash=0.0563 | math=0.5800
  - L=1024 | flash=0.1835 | math=1.9929

- B=8, H=8
  - L=128 | flash=0.0268 | math=0.1593
  - L=256 | flash=0.0358 | math=0.3484
  - L=512 | flash=0.1001 | math=1.0936
  - L=1024 | flash=0.3090 | math=3.7894

- B=8, H=16
  - L=128 | flash=0.0272 | math=0.2253
  - L=256 | flash=0.0590 | math=0.6461
  - L=512 | flash=0.1652 | math=2.0554
  - L=1024 | flash=0.5612 | math=7.4070

- B=16, H=16
  - L=128 | flash=0.0438 | math=0.4302
  - L=256 | flash=0.0997 | math=1.2122
  - L=512 | flash=0.2988 | math=3.9868
  - L=1024 | flash=1.0852 | math=14.7282

## In progress

I am adding implementation with Tensor Cores

