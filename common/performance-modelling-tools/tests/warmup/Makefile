buildtype ?= omp
DEFS = -DUSEOPENMP
OPTFLAGS = -O2
CXXFLAGS = -fPIC -std=c++17 $(OPTFLAGS) -fopenmp
CXX = gcc


ifeq ($(buildtype), cuda)
CXX = nvcc
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++14 $(OPTFLAGS)
DEFS = -DUSECUDA
endif

ifeq ($(buildtype), hip)
CXX = hipcc
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++14 $(OPTFLAGS)
DEFS = -DUSEHIP 
endif

ifeq ($(buildtype), cudaomp)
CXX = nvc++
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++14 $(OPTFLAGS) -mp=gpu -Minfo=mp -gpu=cc70
endif

ifeq ($(buildtype), cudaacc)
CXX = nvc++
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++14 $(OPTFLAGS) -acc -gpu=cc70
#DEFS = -D_OPENACC
endif

ifeq ($(buildtype), cudahip)
CXX = hipcc
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++14 $(OPTFLAGS) 
DEFS = -DUSEHIP
endif

ifeq ($(buildtype), hipomp)
CXX = amdclang++
CXXFLAGS = -std=c++14 -fPIC $(OPTFLAGS) -fopenmp --offload-arch=gfx908 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908
#DEFS =
endif

ifeq ($(buildtype), hipacc)
CXX = amdclang++
CXXFLAGS = -std=c++14 -fPIC $(OPTFLAGS) -fopenacc --offload-arch=gfx908 -fopenacc-targets=amdgcn-amd-amdhsa -march=gfx908
#DEFS =
endif

CXXFLAGS += $(DEFS) 
INCDIR = -I$(shell pwd)/../profile_util/include/ -Iinclude/ 
LDFLAGS = -L$(shell pwd)/../profile_util/lib/
#ifeq ($(buildtype), hip)
#LDFLAGS += -Wl,-rpath=$(shell pwd)/../profile_util/lib/
#endif
# ifeq ($(buildtype), omp)
# LDFLAGS += -Wl,-rpath=$(shell pwd)/../profile_util/lib/
# endif
LIBS = -lprofile_util_$(buildtype)

# Target executable name:
EXE = bin/warm_up_test.$(buildtype).exe

OBJS = obj/main.o \
obj/logger.o \
obj/WarmupGPU.o \

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

ifeq ($(buildtype), cudaomp)
obj/WarmupGPU.o: src/WarmupGPU.cu include/*.h 
	cp src/WarmupGPU.cu src/WarmupGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/WarmupGPU.cpp -o obj/WarmupGPU.o ;
	rm src/WarmupGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), cudaacc)
obj/WarmupGPU.o: src/WarmupGPU.cu include/*.h 
	cp src/WarmupGPU.cu src/WarmupGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/WarmupGPU.cpp -o obj/WarmupGPU.o ;
	rm src/WarmupGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), hipomp)
obj/WarmupGPU.o: src/WarmupGPU.cu include/*.h 
	cp src/WarmupGPU.cu src/WarmupGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/WarmupGPU.cpp -o obj/WarmupGPU.o ;
	rm src/WarmupGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) $(INCDIR) -c $< -o $@
endif
ifeq ($(buildtype), omp)
obj/WarmupGPU.o: src/WarmupGPU.cu include/*.h 
	cp src/WarmupGPU.cu src/WarmupGPU.cpp ;
	$(CXX) $(CXXFLAGS) $(INCDIR) -c src/WarmupGPU.cpp -o obj/WarmupGPU.o ;
	rm src/WarmupGPU.cpp ;
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
	
