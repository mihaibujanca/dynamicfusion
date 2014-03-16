include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)

file(GLOB srcs *.cpp *.hpp)
add_executable(demo ${srcs})
target_link_libraries(demo ${OpenCV_LIBS} kfusion)

set_target_properties(demo PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
install(FILES ${srcs} DESTINATION app COMPONENT main)
