include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)

file(GLOB srcs *.cpp *.hpp)
add_executable(demo ${srcs})
target_link_libraries(demo ${OpenCV_LIBS} ${Boost_LIBRARIES} kfusion)
if(OPENNI_FOUND)
    target_compile_definitions(demo PRIVATE OPENNI_FOUND=1)
endif()

set_target_properties(demo PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
install(FILES ${srcs} DESTINATION app COMPONENT main)
