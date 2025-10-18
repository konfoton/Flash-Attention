NVCC = nvcc
CXX = g++

NVCC_FLAGS = -O3 -std=c++11 -arch=sm_86
CXX_FLAGS = -O3 -std=c++11 -Wall

CUDA_LIBS = -lcudart

NORMAL_CPU_TARGET = normal_cpu
BENCH_TARGET = bench

FLASH_SRC = flash_attention.cu
NORMAL_GPU_SRC = normal_attention.cu
NORMAL_CPU_SRC = normal_attention_cpu.cpp
BENCH_SRC = benchmark.cu

OBJS = flash_attention.o normal_attention.o

all: $(BENCH_TARGET) $(NORMAL_CPU_TARGET)

flash_attention.o: $(FLASH_SRC) attention_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

normal_attention.o: $(NORMAL_GPU_SRC) attention_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(NORMAL_CPU_TARGET): $(NORMAL_CPU_SRC)
	$(CXX) $(CXX_FLAGS) $< -o $@

$(BENCH_TARGET): $(BENCH_SRC) $(OBJS) attention_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(BENCH_SRC) $(OBJS) -o $@

clean:
	rm -f $(NORMAL_CPU_TARGET) $(BENCH_TARGET) $(OBJS)

run_normal_cpu: $(NORMAL_CPU_TARGET)
	./$(NORMAL_CPU_TARGET)

.PHONY: all clean run_normal_cpu run_bench

run_bench: $(BENCH_TARGET)
	./$(BENCH_TARGET)
