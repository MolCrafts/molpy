
message(STATUS "Configuring tests...")

find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    Include(FetchContent)

    FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0 # or a later release
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

file(GLOB test_sources ${PROJECT_SOURCE_DIR}/tests/test_*.cpp)

add_executable(molcpp_test ${test_sources})
target_link_libraries(molcpp_test PRIVATE molcpp)
target_link_libraries(molcpp_test PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(molcpp_test)