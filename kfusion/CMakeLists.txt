set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
add_module_library(kfusion)
if(OPENNI_FOUND)
    target_compile_definitions(kfusion PRIVATE OPENNI_FOUND=1)
endif()

message("CUDA_LIBRARIES:" ${CUDA_LIBRARIES})
message("CUDA_CUDART_LIBRARY:" ${CUDA_CUDART_LIBRARY})
message("CUDA_CUDA_LIBRARY:" ${CUDA_CUDA_LIBRARY})
find_package(Threads REQUIRED)

target_link_libraries(kfusion
        ${OPENMESH_LIBRARIES}
        ${OPT_LIBRARIES}
        ${TERRA_LIBRARIES}
        Threads::Threads
        ${OpenCV_LIBS}
        ${OPENNI_LIBRARY}
        ${CERES_LIBRARIES})

target_include_directories(kfusion PUBLIC ${OPT_INCLUDE_DIRS})
