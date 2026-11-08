# XnYGZip artifact build.
#   make ARCH=sm_120            RTX PRO 6000 Blackwell (paper)
#   make ARCH=sm_80             A100 (paper)
#   make CCCL=<path>            CCCL checkout to use instead of the CUDA toolkit's (paper: CCCL 3.5.0)
NVCC  ?= nvcc
ARCH  ?= sm_120
CCCL  ?=
MINBLOCKS ?= 4

INC   = $(if $(CCCL),-I$(CCCL)/cub -I$(CCCL)/libcudacxx/include -I$(CCCL)/thrust,)
FLAGS = -O3 -arch=$(ARCH) -std=c++20 $(INC) -DXNYGZIP_MINBLOCKS=$(MINBLOCKS)

all: bin/xnygzip bin/xnygzip_bench bin/xnygzip_verify

bin/xnygzip: xnygzip.cu
	@mkdir -p bin
	$(NVCC) $(FLAGS) -o $@ xnygzip.cu

bin/xnygzip_bench: bench/bench.cu bench/common.h xnygzip.cu
	@mkdir -p bin
	$(NVCC) $(FLAGS) -o $@ bench/bench.cu

bin/xnygzip_verify: bench/verify.cu bench/common.h xnygzip.cu
	@mkdir -p bin
	$(NVCC) $(FLAGS) -Xcompiler -fopenmp -o $@ bench/verify.cu

clean:
	rm -rf bin

.PHONY: all clean
