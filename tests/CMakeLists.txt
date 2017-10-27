project(ceres_tests)
add_subdirectory(utils)

include_directories(
        ceres_test PUBLIC
        ${CMAKE_SOURCE_DIR}/kfusion/include
)

add_test(ceres_test ceres_test_warp.cpp)
target_link_libraries(
        ceres_test
        ${CERES_LIBRARIES}
        ${GTEST_BOTH_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        kfusion
)
target_include_directories(ceres_test PUBLIC ${CERES_INCLUDE_DIRS})


add_test(nanoflann_test nanoflann_test.cpp)
add_test(warp_test warp_test.cpp)
target_link_libraries(
        warp_test
        ${OpenCV_LIBS}
        ${OPT_LIBRARIES}
        ${TERRA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${GTEST_BOTH_LIBRARIES}
        kfusion
        kfusion
)