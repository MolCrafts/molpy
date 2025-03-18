cmake_minimum_required(VERSION 3.14)

project(
    molcpp
    VERSION 0.1.0
    DESCRIPTION "cross-platform molecular dynamics data structure and algorithm"
    HOMEPAGE_URL "https://github.com/Roy-Kid/molpy.git"
    LANGUAGES CXX
)

option(MOLCPP_BUILD_TESTS "Build the tests" ON)


add_library(molcpp)

file(GLOB_RECURSE sources ${PROJECT_SOURCE_DIR}/src/*.cpp)

target_sources(molcpp PRIVATE ${sources})

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(molcpp PUBLIC STATIC)
endif()

target_include_directories(
    molcpp ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)


find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

target_include_directories(
    molcpp
    PUBLIC
    ${xtl_INCLUDE_DIRS}
    ${xtensor_INCLUDE_DIRS}
)

target_link_libraries(molcpp PUBLIC xtl xtensor)

target_compile_features(molcpp PUBLIC cxx_std_17)


if (MOLCPP_BUILD_TESTS)
    add_subdirectory(tests)
endif()