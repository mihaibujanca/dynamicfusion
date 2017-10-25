project(opt_tests)

add_test(opt_test opt_test.cpp)

target_link_libraries(
        opt_test
        ${OpenCV_LIBS}
        ${OPT_LIBRARIES}
        ${TERRA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${GTEST_BOTH_LIBRARIES}
        kfusion
)

target_include_directories(
        opt_test PUBLIC
        ${CMAKE_SOURCE_DIR}/kfusion/include/
)
