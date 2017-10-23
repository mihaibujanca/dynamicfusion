project(opt_testing)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

include_directories(include ${OPT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(opt_test mLibSource.cpp opt_test.cpp include/CombinedSolver.h)

find_package(Threads REQUIRED)
message("CUDA CUDART:" ${CUDA_CUDART_LIBRARY})
target_link_libraries(
        opt_test
        ${OPENMESH_LIBRARIES}
        ${OPT_LIBRARIES}
        ${TERRA_LIBRARIES}
        ${GTEST_BOTH_LIBRARIES}
        ${CMAKE_DL_LIBS}
        ${CUDA_CUDART_LIBRARY}
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
