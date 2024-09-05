.PHONY: all clean

LIBDIR=$(shell pwd)/../../lib/
INCDIR=$(shell pwd)/../../include/

CXX=hipcc
ARCH=gfx90a
#CXX=nvc++
#ARCH=unified
#CXX=nvc++


ifeq ($(CXX), hipcc)
GPUFLAGS=-D_HIP -D__HIP_PLATFORM_AMD__ -DGPU_DEBUG -DUSE_SLURM #-DOMP_GPU_OFFLOAD
LIB=-lprofile_util_hip_omp
OMPFLAGS=-DUSEOPENMP -fopenmp --offload-arch=$(ARCH) -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(ARCH)
#OMPFLAGS=-DUSEOPENMP -fopenmp --offload-arch=$(ARCH) 
FLAGS=-std=c++17 -fPIC -I$(INCDIR) -O3 -munsafe-fp-atomics 
FLAGS+=-stdlib++-isystem /opt/cray/pe/gcc/12.2.0/snos/include/g++ -stdlib++-isystem /opt/cray/pe/gcc/12.2.0/snos/include/g++/x86_64-suse-linux
LFLAGS=-L$(LIBDIR)  -Wl,-rpath=$(LIBDIR)
endif

ifeq ($(CXX), nvc++)
GPUFLAGS=-D_CUDA -DGPU_DEBUG -DOMP_GPU_OFFLOAD
LIB=-lprofile_util_cuda_omp
OMPFLAGS=-mp
FLAGS=-std=c++17 -fPIC -I$(INCDIR) -cuda -gpu=$(ARCH) -O3
ifeq ($(ARCH), unified)
FLAGS+=-DGPU_UNIFIED
endif
FLAGS+=#-DKERNEL2 #-DRUN_OPENMP_WITH_GPU_KERNELS
LFLAGS=-L$(LIBDIR) -Wl,-rpath=$(LIBDIR)
endif 

ifeq ($(CXX), nvcc)
GPUFLAGS=-D_CUDA
LIB=-lprofile_util_cuda_omp
OMPFLAGS=-mp
FLAGS=-Xcompiler -std=c++17 -Xcompiler -fPIC -I$(INCDIR) 
LFLAGS=-L$(LIBDIR) -Xcompiler -Wl,-rpath=$(LIBDIR)
endif

all: bin/gpu-openmp

clean:
	rm -f bin/gpu-openmp
	rm -f obj/*.o

bin/gpu-openmp: src/gpu-openmp.cpp src/kernels.h src/kernels.cu src/common.h
	$(CXX) $(GPUFLAGS) $(FLAGS) -c src/kernels.cu -o obj/kernels.o
	$(CXX) $(GPUFLAGS) $(FLAGS) $(OMPFLAGS) -c src/gpu-openmp.cpp -o obj/gpu-openmp.o
	$(CXX) $(GPUFLAGS) $(FLAGS) -fopenmp obj/*.o -o bin/gpu-openmp $(LFLAGS) $(LIB)

