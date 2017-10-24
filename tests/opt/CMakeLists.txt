project(opt_testing)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

include_directories(include ${OPT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

add_executable(opt_test opt_test.cpp)

find_package(Threads REQUIRED)

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
        ${CMAKE_SOURCE_DIR}/kfusion/include/
)
