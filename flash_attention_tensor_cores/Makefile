ARCH ?= sm_86
NVCC_FLAGS = -O3 --std=c++17 -arch=$(ARCH) -rdc=true

all: benchmark_tensor_flash_attention

# final: test.o flash.o
# 	nvcc $(NVCC_FLAGS) test.o flash.o -o final

benchmark_tensor_flash_attention: benchmark_tensor_flash_attention.o flash.o
	nvcc $(NVCC_FLAGS) benchmark_tensor_flash_attention.o flash.o -o benchmark_tensor_flash_attention

benchmark_tensor_flash_attention.o: benchmark_tensor_flash_attention.cu
	nvcc $(NVCC_FLAGS) -c benchmark_tensor_flash_attention.cu -o benchmark_tensor_flash_attention.o

# test.o: testing.cu
# 	nvcc $(NVCC_FLAGS) -arch=sm_86 -c testing.cu -o test.o

flash.o: tensor_flash_attention.cu tensor_flash_attention.cuh cuda_utils.cuh
	nvcc $(NVCC_FLAGS) -c tensor_flash_attention.cu -o flash.o

clean:
	rm -f *.o benchmark_tensor_flash_attention final benchmark_tensor_flash_attention