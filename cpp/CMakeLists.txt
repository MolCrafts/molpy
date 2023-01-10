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

# Install
# install(TARGETS molpy_cpp DESTINATION lib)

# Install the headers
# install(FILES graph.h DESTINATION include)

# Create base directory
# install(DIRECTORY molpy_cpp DESTINATION include)