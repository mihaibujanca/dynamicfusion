include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)

#Adding CombinedSolver because the IDE provides code insight. Not actually needed.
add_executable(dynamicfusion demo.cpp ../kfusion/include/opt/CombinedSolver.h)
target_link_libraries(dynamicfusion ${OpenCV_LIBS}
        ${OPT_LIBRARIES}
        ${TERRA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        kfusion
        )

set_target_properties(dynamicfusion PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


#if(OPENNI_FOUND)
#    add_executable(dynamicfusion_kinect dynamicfusion_kinect.cpp)
#    target_link_libraries(dynamicfusion_kinect ${OPENNI_LIBRARY} ${OpenCV_LIBS} kfusion)
#
#    set_target_properties(dynamicfusion_kinect PROPERTIES
#        DEBUG_POSTFIX "d"
#        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
#        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
#    install(TARGETS dynamicfusion_kinect RUNTIME DESTINATION bin COMPONENT main)
#endif()

install(TARGETS dynamicfusion RUNTIME DESTINATION bin COMPONENT main)
