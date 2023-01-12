cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(molpy_cpp VERSION 0.1.0)

# Src
AUX_SOURCE_DIRECTORY(. SRC_FILES)

# Headers
# set(PROJECT_SOURCE_DIR ".")
# set(PROJECT_INCLUDE_DIR ".")

# Source files
set(SOURCE_FILES
    graph.h
    graph.cpp
)

# Set up such that XCode organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

# Add library
add_library(molpy_cpp SHARED ${SOURCE_FILES})

# Include directories
target_include_directories(molpy_cpp PRIVATE .)

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -Wall -O0")
    message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")

    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )    

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    set(TEST_FILES
    tests/test_graph.cpp
    )    
    include_directories(.)
    include_directories(/usr/include/python3.8)
    enable_testing()

    add_executable(
    molpy_cpp_test
    ${TEST_FILES}
    )
    target_link_libraries(
        molpy_cpp_test
    GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(molpy_cpp_test)

endif()

# Install
# install(TARGETS molpy_cpp DESTINATION lib)

# Install the headers
# install(FILES graph.h DESTINATION include)

# Create base directory
# install(DIRECTORY molpy_cpp DESTINATION include)