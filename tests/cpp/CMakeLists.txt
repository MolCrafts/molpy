include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

message("Testing: ${PROJECT_SOURCE_DIR}")

add_executable(
  molpy_cpp_test
  test_random_walk.cc
)
target_link_libraries(
  molpy_cpp_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(molpy_cpp_test)