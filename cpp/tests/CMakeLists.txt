cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(molpy_test)

#find_library(molpy_lib molpy_cpp HINTS ../../lib)
find_package(pybind11 REQUIRED)
include_directories(../)
include_directories(/usr/include/python3.8)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Source files
set(TEST_FILES
    test_graph.cpp
)

# if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
#     set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -Wall -O0")
#     message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
#     add_executable(test ${TEST_FILES})

# elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
#     set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")
#     message("Release mode:${CMAKE_C_FLAGS_RELEASE}")
#     add_executable(test ${TEST_FILES})
# else()
#     message("else:${CMAKE_BUILD_TYPE}")
#     message("else:${CMAKE_C_FLAGS_RELEASE}")
#     add_executable(test ${TEST_FILES})
# endif()

enable_testing()

add_executable(
  molpy_cpp_test
  test_graph.cpp
)
target_link_libraries(
    molpy_cpp_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(molpy_cpp_test)

# target_link_libraries(test PUBLIC ${molpy_lib})