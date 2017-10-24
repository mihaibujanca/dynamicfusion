set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
add_module_library(kfusion)
if(OPENNI_FOUND)
    target_compile_definitions(kfusion PRIVATE OPENNI_FOUND=1)
endif()

message("TERRA:" ${TERRA_LIBRARIES})
message("Opt:" ${OPT_LIBRARIES})
message("OPENMESH_LIBRARIES:" ${OPENMESH_LIBRARIES})
target_link_libraries(kfusion ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${OPENMESH_LIBRARIES}
        ${OPT_LIBRARIES}
        ${TERRA_LIBRARIES}
        ${OpenCV_LIBS}
        ${OPENNI_LIBRARY}
        ${CERES_LIBRARIES}
        Threads::Threads
        )
target_include_directories(kfusion PUBLIC ${OPT_INCLUDE_DIRS})
