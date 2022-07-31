cmake_minimum_required(VERSION 3.15)

project(molpy_cpp VERSION "0.0.1")

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# project settings
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # 添加c++11标准支持

# set(CMAKE_C_COMPILER "/usr/bin/gcc") # 默认c编译器
# set(CMAKE_CXX_COMPILER "/usr/bin/g++") # 默认c++编译器

set(CMAKE_BUILD_TYPE "Debug")  # Debug模式 选项: Release Debug MinSizeRel RelWithDebInfo

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  # debug模式下 gdb相关选项
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") # release模式下 gdb相关选项

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin") # 可执行文件输出目录 deprecated

set(TEST "FALSE" CACHE STRING "is start to test raw cpp code without python")

include_directories("${PROJECT_SOURCE_DIR}/include") # 包含目录

set(CMAKE_CXX_STANDARD 14)

# 这段代码可以区分操作系统
MESSAGE(">>> Identifying the OS...")
if(WIN32)
  MESSAGE(">>> Windows")
elseif(APPLE)
  MESSAGE(">>> MacOS")
elseif(UNIX)
  MESSAGE(">>> Linux")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

if(TEST)

    find_package(GTest REQUIRED)
    if(${GTest_FOUND})
        message(">>> GTest is found")
        include_directories(${GTEST_INCLUDE_DIRS})

    else()
        message(">>> GTest is not found")
        message(">>> Installing GTest")
        include(FetchContent)
        FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    include(GoogleTest)
    enable_testing()
    add_subdirectory(tests)

endif()

# build molpy_cpp

aux_source_directory(src/modeller MODELLER)

pybind11_add_module(
    molpy_cpp
    src/molpy_cpp.cpp
    ${MODELLER}
)

target_compile_definitions(molpy_cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS molpy_cpp DESTINATION .)
