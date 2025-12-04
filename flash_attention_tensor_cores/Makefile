ARCH ?= sm_86
NVCC_FLAGS = -O3 --std=c++17 -arch=$(ARCH) -rdc=true

all: benchmark_tensor_flash_attention testing

benchmark_tensor_flash_attention: benchmark_tensor_flash_attention.o flash.o wrapper.o
	nvcc $(NVCC_FLAGS) benchmark_tensor_flash_attention.o flash.o wrapper.o -o benchmark_tensor_flash_attention

benchmark_tensor_flash_attention.o: benchmark_tensor_flash_attention.cu
	nvcc $(NVCC_FLAGS) -c benchmark_tensor_flash_attention.cu -o benchmark_tensor_flash_attention.o

testing: testing.o wrapper.o flash.o
	nvcc $(NVCC_FLAGS) testing.o flash.o wrapper.o -o testing

testing.o: testing.cu
	nvcc $(NVCC_FLAGS) -c testing.cu -o testing.o

flash.o: tensor_flash_attention.cu tensor_flash_attention.cuh cuda_utils.cuh
	nvcc $(NVCC_FLAGS) -c tensor_flash_attention.cu -o flash.o

wrapper.o: wrapper.cu wrapper.cuh tensor_flash_attention.cuh
	nvcc $(NVCC_FLAGS) -c wrapper.cu -o wrapper.o

clean:
	rm -f *.o benchmark_tensor_flash_attention testing final benchmark_tensor_flash_attention