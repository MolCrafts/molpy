cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(molpy_test)

find_library(molpy_lib molpy_cpp HINTS ../../lib)
include_directories(../)

# Source files
set(TEST_FILES
    test_graph.cpp
)

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -Wall -O0")
    message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
    add_executable(test ${TEST_FILES})

elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")
    message("Release mode:${CMAKE_C_FLAGS_RELEASE}")
    add_executable(test ${TEST_FILES})
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_C_FLAGS_RELEASE}")
    add_executable(test ${TEST_FILES})
endif()

target_link_libraries(test PUBLIC ${molpy_lib})