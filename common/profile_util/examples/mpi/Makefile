.PHONY: all clean

LIBDIR=$(shell pwd)/../../lib/
LIB=-lprofile_util_mpi_omp
INCDIR=$(shell pwd)/../../include/ 
FLAGS=-L$(LIBDIR) -I$(INCDIR) -Wl,-rpath=$(LIBDIR) -D_MPI -std=c++17 -fopenmp
CXXFLAGS?=

all: bin/mpivec_cpp bin/mpicomm_cpp bin/mpi-compute bin/mpi-io

clean:
	rm -f bin/mpivec_cpp
	rm -f bin/mpicomm_cpp
	rm -f bin/mpi-compute
	rm -f bin/mpi-io

bin/mpivec_cpp: src/mpivec.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) src/mpivec.cpp -o bin/mpivec_cpp $(LIB)

bin/mpicomm_cpp: src/mpicomm.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) src/mpicomm.cpp -o bin/mpicomm_cpp $(LIB)

bin/mpi-compute: src/mpi-compute.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) src/mpi-compute.cpp -o bin/mpi-compute $(LIB)

bin/mpi-io: src/mpi-io.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) src/mpi-io.cpp -o bin/mpi-io $(LIB)
