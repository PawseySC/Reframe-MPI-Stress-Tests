buildtype ?= omp
DEFS = -DRUNWITHOUTALLOC
OPTFLAGS = -O2
CXXFLAGS = -fPIC -std=c++17 $(OPTFLAGS) -fopenmp
CXX = gcc


ifeq ($(buildtype), cuda)
CXX = nvcc
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++17 $(OPTFLAGS)
DEFS = -DUSECUDA -DRUNWITHOUTALLOC
endif

ifeq ($(buildtype), hip)
CXX = hipcc
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++17 $(OPTFLAGS)
DEFS = -DUSEHIP -DRUNWITHOUTALLOC
endif

ifeq ($(buildtype), nvomp)
CXX = nvc++
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++17 $(OPTFLAGS) -mp=gpu -Minfo=mp -gpu=cc70
endif

ifeq ($(buildtype), gccomp)
CXX = g++
CXXFLAGS = -fPIC -std=c++17 $(OPTFLAGS) -fopenmp
endif

ifeq ($(buildtype), nvacc)
CXX = nvc++
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++17 $(OPTFLAGS) -acc -gpu=cc70
endif

ifeq ($(buildtype), nvompacc)
CXX = nvc++
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++17 $(OPTFLAGS) -mp -Minfo=mp -acc -gpu=cc70
endif

ifeq ($(buildtype), gccacc)
CXX = g++
CXXFLAGS = -fPIC -std=c++17 $(OPTFLAGS) -fopenacc
endif

ifeq ($(buildtype), cudahip)
CXX = hipcc
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++17 $(OPTFLAGS) 
DEFS = -DUSEHIP -DRUNWITHOUTALLOC
endif

ifeq ($(buildtype), hipomp)
#CXX = /opt/rocm/bin/amdclang++
CXXFLAGS = -std=c++17 -fPIC $(OPTFLAGS) -fopenmp --offload-arch=gfx90a -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
CXX = CC
#CXXFLAGS = -std=c++17 -fPIC $(OPTFLAGS) -fopenmp 
#--offload-arch=gfx90a -fopenmp-targets=amdgcn-amd-amdhsa
#--offload-arch=gfx90a -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
#DEFS =
endif

ifeq ($(buildtype), hipacc)
CXX = CC
CXXFLAGS = -std=c++17 -fPIC $(OPTFLAGS) -fopenacc 
#--offload-arch=gfx90a -fopenacc-targets=amdgcn-amd-amdhsa -march=gfx90a
#DEFS =
endif

CXXFLAGS += $(DEFS) 
INCDIR = -I$(shell pwd)/../profile_util/include/ -Iinclude/ 
LDFLAGS = -L$(shell pwd)/../profile_util/lib/
LIBS = -lprofile_util_$(buildtype)

# Target executable name:
EXE = bin/multigpu.$(buildtype).exe

OBJS = obj/main.o \
obj/logger.o \
obj/multiGPU.o \

.PHONY: libprofile_util $(EXE)

all : libprofile_util $(EXE) 

BLUE='\034[0;31m'
NC='\033[0m' # No Color

libprofile_util:
	@echo "Making profile_util library" ;
	$(MAKE) -C ../profile_util/ COMPILER=$(CXX) CXXFLAGS="$(CXXFLAGS)"; 
	cp ../profile_util/lib/libprofile_util.so ../profile_util/lib/libprofile_util_$(buildtype).so 

$(EXE) : $(OBJS)
	@echo "Making $(EXE) " ;
	$(CXX) $(CXXFLAGS) $(INCDIR) $(OBJS) $(LDFLAGS) -o $@ $(LIBS)
	@echo "To run code ensure that you add the following to the environment"
	@echo -e "export LD_LIBRARY_PATH=$(shell pwd)/../profile_util/lib/:\044LD_LIBRARY_PATH"

ifeq ($(buildtype), nvomp)
obj/multiGPU.o: src/multiGPU.cu include/*.h 
	cp src/multiGPU.cu src/multiGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/multiGPU.cpp -o obj/multiGPU.o ;
	rm src/multiGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), gccomp)
obj/multiGPU.o: src/multiGPU.cu include/*.h 
	cp src/multiGPU.cu src/multiGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/multiGPU.cpp -o obj/multiGPU.o ;
	rm src/multiGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), nvacc)
obj/multiGPU.o: src/multiGPU.cu include/*.h 
	cp src/multiGPU.cu src/multiGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/multiGPU.cpp -o obj/multiGPU.o ;
	rm src/multiGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), nvompacc)
obj/multiGPU.o: src/multiGPU.cu include/*.h 
	cp src/multiGPU.cu src/multiGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/multiGPU.cpp -o obj/multiGPU.o ;
	rm src/multiGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), gccacc)
obj/multiGPU.o: src/multiGPU.cu include/*.h 
	cp src/multiGPU.cu src/multiGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/multiGPU.cpp -o obj/multiGPU.o ;
	rm src/multiGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), hipomp)
obj/multiGPU.o: src/multiGPU.cu include/*.h 
	cp src/multiGPU.cu src/multiGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/multiGPU.cpp -o obj/multiGPU.o ;
	rm src/multiGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), omp)
obj/multiGPU.o: src/multiGPU.cu include/*.h 
	cp src/multiGPU.cu src/multiGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/multiGPU.cpp -o obj/multiGPU.o ;
	rm src/multiGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), cuda)
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
obj/%.o: src/%.cu include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), cudahip)
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
obj/%.o: src/%.cu include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), hip)
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
obj/%.o: src/%.cu include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif



clean:
	rm -f $(OBJS) $(EXE)
	$(MAKE) -C ../profile_util/ clean  
	
