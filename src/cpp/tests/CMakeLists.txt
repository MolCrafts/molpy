
message(STATUS "Configuring tests...")

find_package(Catch2 3 QUIET)

file(GLOB test_sources ${PROJECT_SOURCE_DIR}/tests/test_*.cpp)

add_executable(molcpp_test ${test_sources})
target_link_libraries(molcpp_test PRIVATE molcpp)
target_link_libraries(molcpp_test PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(molcpp_test)