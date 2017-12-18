EXECUTABLE = shape_from_shading
OBJS = build/CUDAImageSolver.o build/mLibSource.o build/main.o build/SFSSolver.o  build/SimpleBuffer.o


UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -L../external/OpenMesh/lib/osx -Wl,-rpath,../external/OpenMesh/lib/osx
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -L../external/OpenMesh/lib/linux -Wl,-rpath,../external/OpenMesh/lib/linux
endif

LFLAGS += -lOpenMeshCore -lOpenMeshTools


include ../shared/make_template.inc
