
message(">>> Start to test raw c++ code without python wrapper")

aux_source_directory(${PROJECT_SOURCE_DIR}/src/modeller MODELLER)

add_executable(
    molpy_cpp_test
    ${MODELLER}
    test_cpp/test_random_walk.cpp
    test_cpp/main.cpp
    )
target_link_libraries(
    molpy_cpp_test
    )      
