.PHONY: all clean

GPU=hip
MPICXX=hipcc
LIBDIR=$(shell pwd)/common/profile_util/lib/
LIB=-lprofile_util_${GPU}_mpi_omp -lmpi -lmpi_gtl_hsa
INCDIR=$(shell pwd)/common/profile_util/include/ 
FLAGS=-L$(LIBDIR) -I$(INCDIR) -Wl,-rpath=$(LIBDIR) -D_HIP -D_MPI -std=c++17 -fopenmp 
CXXFLAGS?=-I${CRAY_MPICH_DIR}/include/ -L${CRAY_MPICH_DIR}/lib/ -L${CRAY_MPICH_ROOTDIR}/gtl/lib/

all: gpu-mpi-comms

clean:
	rm -f gpu-mpi-comms


gpu-mpi-comms: gpu-mpi-comms.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) gpu-mpi-comms.cpp -o gpu-mpi-comms $(LIB)
