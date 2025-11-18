NVCC_FLAGS = -g -G -lineinfo -O0 --std=c++17

all: final

final: test.o flash.o
	nvcc $(NVCC_FLAGS) test.o flash.o -o final

test.o: testing.cu
	nvcc $(NVCC_FLAGS) -arch=sm_86 -c testing.cu -o test.o

flash.o: tensor_flash_attention.cu
	nvcc $(NVCC_FLAGS) -arch=sm_86 -c tensor_flash_attention.cu -o flash.o

clean:
	rm -f test.o flash.o final