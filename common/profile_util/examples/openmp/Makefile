.PHONY: all clean

F90=gfortran 
OMPFLAGS=-fopenmp -DUSEOPENMP 
#--offload-arch=gfx90a -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
FOMPFLAGS=-fopenmp 
LIBDIR=$(shell pwd)/../../lib/
LIB=-lprofile_util_omp
#LIB=-lprofile_util_hip_omp
INCDIR=$(shell pwd)/../../include/
FLAGS=-L$(LIBDIR) -I$(INCDIR) -Wl,-rpath=$(LIBDIR)

all: bin/openmpvec_cpp bin/openmpvec_c bin/openmpvec_fort

clean:
	rm -f bin/openmpvec_cpp bin/openmpvec_c bin/openmpvec_fort

bin/openmpvec_cpp: src/openmpvec.cpp
	$(CXX) $(OMPFLAGS) $(FLAGS) src/openmpvec.cpp -o bin/openmpvec_cpp $(LIB) 

bin/openmpvec_c: src/openmpvec.c
	$(CXX) $(OMPFLAGS) $(FLAGS) src/openmpvec.c -o bin/openmpvec_c $(LIB)

bin/openmpvec_fort: src/openmpvec.f90
	$(F90) $(FOMPFLAGS) $(FLAGS) src/openmpvec.f90 -o bin/openmpvec_fort $(LIB)

	
