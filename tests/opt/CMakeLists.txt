add_executable(opt_test WarpingSolver.cu CUDAWarpingSolver.cpp LandMark.cpp LandMarkSet.cpp opt_test.cpp mLibSource.cpp)
find_package(Threads REQUIRED)
target_link_libraries(
        opt_test
        /usr/local/cuda/lib64/libcudart.so # FIXME: hardcoded library path because ${CUDA_LIBRARIES} links statically
        ${OPT_LIBRARIES}
        ${TERRA_LIBRARIES}
        ${GTEST_BOTH_LIBRARIES}
        ${CMAKE_DL_LIBS}
        ${OPENMESH_LIBRARIES}
        Threads::Threads
        kfusion
)

target_include_directories(
        opt_test PUBLIC
        ${OPT_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/kfusion/include/
        include
)
