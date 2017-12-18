CXX = clang++
LINK = clang++
NVCC = /usr/local/cuda/bin/nvcc
SRC ?= src
CUDAHOME += /usr/local/cuda
TERRAHOME ?= ../../../terra
TERRA ?= ../../../terra/release/terra

ifeq ($(UNAME), Linux)
  S4 = 64
endif

FLAGS += -O3 -g -std=c++11 -I$(SRC) -I$(SRC)/cutil/inc -I../../API/release/include -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I../external/mLib/include -I../external -I../external/OpenMesh/include
NVCCFLAGS += -arch sm_30 $(FLAGS)
LFLAGS += -std=c++11 -L../../API/release/lib -L$(CUDAHOME)/lib$(S4) -lcudart -L$(TERRAHOME)/lib -lterra -ldl
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -Wl,-rpath,$(CUDAHOME)/lib$(S4) -pthread
  S4 = 64
endif

.SUFFIXES:
.PHONY:	all purge clean

all: $(EXECUTABLE)

OPTLIB = ../../API/release/lib/libOptDev.a

build/%.o:	$(SRC)/%.cpp
	@mkdir -p build
	$(CXX) $(FLAGS) -MP -MD $< -c -o $@
build/%.o:	$(SRC)/%.cu
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(EXECUTABLE):	$(OBJS) $(OPTLIB)
	$(CXX) $^ -o $@ $(LFLAGS)

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)
