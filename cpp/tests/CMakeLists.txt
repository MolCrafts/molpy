cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(molpy_test)

find_library(molpy_lib molpy_cpp HINTS ../../lib)
include_directories(../../include)

# Source files
set(TEST_FILES
    test_graph.cpp
)

add_executable(test test_graph.cpp)

target_link_libraries(test PUBLIC ${molpy_lib})