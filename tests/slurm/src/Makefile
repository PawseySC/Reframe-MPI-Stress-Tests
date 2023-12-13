.PHONY: all clean

CXX=CC
MPICXX=CC
LIBDIR=$(shell pwd)/common/profile_util/lib/
LIB=-lprofile_util_mpi_omp
INCDIR=$(shell pwd)/common/profile_util/include/
FLAGS=-L$(LIBDIR) -I$(INCDIR) -Wl,-rpath=$(LIBDIR) -D_MPI -std=c++17
CXXFLAGS?=-fopenmp

all: mpicomm affinity_report.nompi

clean: 
	rm -f mpicomm affinity_report.nompi

mpicomm: mpicomm.cpp
	$(MPICXX) $(FLAGS) $(CXXFLAGS) mpicomm.cpp -o mpicomm $(LIB)

affinity_report.nompi: affinity_report.nompi.cpp
	$(CXX) $(FLAGS) $(CXXFLAGS) affinity_report.nompi.cpp -o affinity_report.nompi $(LIB)
