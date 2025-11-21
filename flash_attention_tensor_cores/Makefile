NVCC_FLAGS = -O3 --std=c++17

all: final benchmark_tensor_flash_attention

final: test.o flash.o
	nvcc $(NVCC_FLAGS) test.o flash.o -o final

benchmark_tensor_flash_attention: benchmark_tensor_flash_attention.cu flash.o
	nvcc $(NVCC_FLAGS) -arch=sm_86 benchmark_tensor_flash_attention.cu flash.o -o benchmark_tensor_flash_attention

test.o: testing.cu
	nvcc $(NVCC_FLAGS) -arch=sm_86 -c testing.cu -o test.o

flash.o: tensor_flash_attention.cu
	nvcc $(NVCC_FLAGS) -arch=sm_86 -c tensor_flash_attention.cu -o flash.o

clean:
	rm -f test.o flash.o final