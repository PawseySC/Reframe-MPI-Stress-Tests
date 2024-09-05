.PHONY: all clean

LIBDIR=$(shell pwd)/../../lib/
INCDIR=$(shell pwd)/../../include/

# GPU=hip
# MPICXX=hipcc
# ARCH=gfx90a
# CRAY=true

GPU=nvidia
CXX=nvc++
ARCH=unified
CRAY=false

ifeq ($(CXX), hipcc)
MPICXX=hipcc
GPUFLAGS=-D_HIP -D__HIP_PLATFORM_AMD__ 
LIB=-lprofile_util_hip_mpi_omp -lmpi
ifeq ($(CRAY),true)
LIB+=-lmpi_gtl_hsa
CXXFLAGS=-I${CRAY_MPICH_DIR}/include/ -L${CRAY_MPICH_DIR}/lib/ -L${CRAY_MPICH_ROOTDIR}/gtl/lib/
endif

OMPFLAGS=-DUSEOPENMP -fopenmp --offload-arch=$(ARCH) -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$(ARCH)
LFLAGS=-L$(LIBDIR) -Wl,-rpath=$(LIBDIR)
FLAGS=-std=c++17 -fPIC -D_MPI -I$(INCDIR) 
endif

ifeq ($(CXX), nvc++)
MPICXX=mpic++
GPUFLAGS=-D_CUDA
OMPFLAGS=-mp
LIB=-lprofile_util_cuda_mpi_omp -lmpi #-lmpi_cxx
LFLAGS=-L$(LIBDIR) -Wl,-rpath=$(LIBDIR)
FLAGS=-std=c++17 -fPIC -D_MPI -I$(INCDIR) -cuda -gpu=$(ARCH) 
endif 


all: bin/gpu-mpi-comm

clean:
	rm -f bin/gpu-mpi-comm
	rm -f obj/*.o


bin/gpu-mpi-comm: src/gpu-mpi-comm.cpp
	$(MPICXX) $(GPUFLAGS) $(FLAGS) $(CXXFLAGS) $(OMPFLAGS) -c src/gpu-mpi-comm.cpp -o obj/gpu-mpi-comm.o
	$(MPICXX) $(GPUFLAGS) $(FLAGS) $(cXXFLAGS) $(OMPFLAGS) obj/*.o -o bin/gpu-mpi-comm $(LFLAGS) $(LIB)

