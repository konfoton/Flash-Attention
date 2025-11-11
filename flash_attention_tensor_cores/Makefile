
all: final

final: test.o flash.o
	nvcc test.o flash.o -o final

test.o: testing.cu
	nvcc -arch=sm_86 -c testing.cu -o test.o

flash.o: tensor_flash_attention.cu
	nvcc -arch=sm_86 -c tensor_flash_attention.cu -o flash.o

clean:
	rm -f test.o flash.o final