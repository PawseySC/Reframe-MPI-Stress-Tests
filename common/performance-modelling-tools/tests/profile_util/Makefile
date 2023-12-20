# Simple Makefile

OUTPUTFILEBASE=libprofile_util
CXXFLAGS = -fPIC -std=c++17 -O2
COMPILER ?=$(CXX)
bt ?= omp

EXTRAFLAGS ?= 
COMPILERFLAGS = $(CXXFLAGS) $(EXTRAFLAGS)

BUILDTYPE ?= serial
DEVICETYPE= cpu  
BUILDNAME ?=

OBJS = obj/mem_util.o obj/time_util.o obj/thread_affinity_util.o 
LIB = lib/$(OUTPUTFILEBASE)$(BUILDNAME)

ifeq ($(bt), hip)
$(LIB).so: $(OBJS)
	@echo "Making $(BUILDTYPE) for $(DEVICETYPE) library"
	$(COMPILER) --offload-arch=gfx90a -shared $(OBJS) -o $(LIB).so
	rm $(OBJS)
endif
ifeq ($(bt), hipomp)
$(LIB).so: $(OBJS)
	@echo "Making $(BUILDTYPE) for $(DEVICETYPE) library"
	$(COMPILER) --offload-arch=gfx90a -shared $(OBJS) -o $(LIB).so
	rm $(OBJS)
endif
ifeq ($(bt), hipacc)
$(LIB).so: $(OBJS)
	@echo "Making $(BUILDTYPE) for $(DEVICETYPE) library"
	$(COMPILER) --offload-arch=gfx90a -shared $(OBJS) -o $(LIB).so
	rm $(OBJS)
endif
ifeq ($(bt), omp)
$(LIB).so: $(OBJS)
	@echo "Making $(BUILDTYPE) for $(DEVICETYPE) library"
	$(COMPILER) -shared $(OBJS) -o $(LIB).so
	rm $(OBJS)
endif
ifeq ($(bt), acc)
$(LIB).so: $(OBJS)
	@echo "Making $(BUILDTYPE) for $(DEVICETYPE) library"
	$(COMPILER) -shared $(OBJS) -o $(LIB).so
	rm $(OBJS)
endif

$(OBJS): obj/%.o : src/%.cpp include/profile_util.h

obj/%.o: src/%.cpp include/profile_util.h
	$(COMPILER) $(COMPILERFLAGS) -Iinclude/ -c $< -o $@

clean:
	rm -f $(LIB).so $(OBJS)
