EXECUTABLE = cotangent_mesh_smoothing
OBJS = build/mLibSource.o build/main.o 

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  LFLAGS += -L../external/OpenMesh/lib/osx -Wl,-rpath,../external/OpenMesh/lib/osx
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -L../external/OpenMesh/lib/linux -Wl,-rpath,../external/OpenMesh/lib/linux
endif

LFLAGS += -lOpenMeshCore -lOpenMeshTools

include ../shared/make_template.inc
