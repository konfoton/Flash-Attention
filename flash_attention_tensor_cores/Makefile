NVCC_FLAGS = -O0 --std=c++17

all: final benchmark

final: test.o flash.o
	nvcc $(NVCC_FLAGS) test.o flash.o -o final

benchmark: benchmark.o flash.o
	nvcc $(NVCC_FLAGS) benchmark.o flash.o -o benchmark_tensor_flash_attention

test.o: testing.cu
	nvcc $(NVCC_FLAGS) -arch=sm_86 -c testing.cu -o test.o

benchmark.o: benchmark_tensor_flash_attention.cu
	nvcc $(NVCC_FLAGS) -arch=sm_86 -c benchmark_tensor_flash_attention.cu -o benchmark.o

flash.o: tensor_flash_attention.cu
	nvcc $(NVCC_FLAGS) -arch=sm_86 -c tensor_flash_attention.cu -o flash.o

clean:
	rm -f test.o flash.o benchmark.o final benchmark_tensor_flash_attention