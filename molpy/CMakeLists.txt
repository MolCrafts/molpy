cmake_minimum_required(VERSION 3.0.0)
project(molpy_kernel VERSION 0.1.0)

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SOURCES src/random_park.cpp
           src/simpleRandomWalk.cpp)

find_package(pybind11 REQUIRED)

pybind11_add_module(${PROJECT_NAME} MODULE ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src)

