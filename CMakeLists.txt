cmake_minimum_required(VERSION 2.8.8)

# ---[ Configurations types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
if (DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

find_package(Doxygen)
if(DOXYGEN_FOUND)

  set(DOXYGEN_INPUT ${SOURCE_FILES})
  set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  add_custom_command(
          OUTPUT ${DOXYGEN_OUTPUT}
          COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
          COMMAND ${CMAKE_COMMAND} -E echo "Done."
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS ${DOXYGEN_INPUT}
  )

  add_custom_target(apidoc ALL DEPENDS ${DOXYGEN_OUTPUT})

  add_custom_target(apidoc_forced
          COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
          COMMAND ${CMAKE_COMMAND} -E echo "Done."
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          )

endif(DOXYGEN_FOUND)


# ---[ Solution name
project(kfusion C CXX)

# ---[ utility
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)

# ---[ find dependencies
find_package(OpenCV 2.4.8 REQUIRED COMPONENTS core viz highgui calib3d)
find_package(CUDA 5.0 REQUIRED)
find_package(OpenNI REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENNI_INCLUDE_DIR} "kfusion/src/utils" "kfusion/include/nanoflann")

# ---[ misc settings
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---[ cuda settings
set(HAVE_CUDA 1)
#list(APPEND CUDA_NVCC_FLAGS "-arch;compute_20")
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50")

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

warnings_disable(CMAKE_CXX_FLAGS /wd4985)

add_subdirectory(kfusion)
add_subdirectory(apps)
SET(BUILD_TESTS ON)
if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  if( GTEST_FOUND )
    message( "Found Gtest at ${GTEST_ROOT}")
    message( "GTest Libs: ${GTEST_BOTH_LIBRARIES}")
    message( "GTest Include: ${GTEST_INCLUDE_DIRS}")
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(tests)
  endif()
endif()