project(tests)
add_subdirectory(utils)
add_subdirectory(opt)
include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)

add_executable(ceres_test ceres_test_warp.cpp)
target_link_libraries(
        ceres_test
        ${CERES_LIBRARIES}
        ${GTEST_BOTH_LIBRARIES}
        kfusion
)

target_include_directories(
        ceres_test PUBLIC
        ${CERES_INCLUDE_DIRS}
)

add_executable(nanoflann_test nanoflann_test.cpp)
