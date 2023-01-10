
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(molpy_cpp)

include_directories("${CMAKE_SOURCE_DIR}/cpp/")

file (GLOB SOURCE_FILES "cpp/*.cpp")
file (GLOB HEADER_FILES "cpp/*.h")
# file (GLOB PYTHON_FILES "python/*.cpp" "python/*.hpp")

# Set up such that XCode organizes the files
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES} )

find_package(pybind11 REQUIRED)
pybind11_add_module(molpy_cpp 
  ${SOURCE_FILES}
  ${HEADER_FILES}
	# ${PYTHON_FILES}
)

target_link_libraries(molpy_cpp PUBLIC)

set(PYTHON_LIBRARY_DIR .)

install(TARGETS molpy_cpp
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )