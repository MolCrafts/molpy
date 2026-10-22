cmake_minimum_required(VERSION 3.14)
project(molc_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Path to molrs workspace root (two levels up from tests/cpp/)
set(MOLRS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(MOLRS_CAPI_ROOT "${MOLRS_ROOT}")

# Cargo build profile
set(CARGO_PROFILE "debug" CACHE STRING "Cargo profile: debug or release")
if(CARGO_PROFILE STREQUAL "release")
    set(MOLRS_CARGO_PROFILE_ARGS --release)
elseif(NOT CARGO_PROFILE STREQUAL "debug")
    message(FATAL_ERROR "CARGO_PROFILE must be debug or release")
endif()
# All workspace roots in this repo share one <repo>/target via the committed
# .cargo/config.toml. Respect a local Cargo override for isolated verification.
if(DEFINED ENV{CARGO_TARGET_DIR} AND NOT "$ENV{CARGO_TARGET_DIR}" STREQUAL "")
    set(MOLRS_CARGO_TARGET_DEFAULT "$ENV{CARGO_TARGET_DIR}")
else()
    set(MOLRS_CARGO_TARGET_DEFAULT "${MOLRS_ROOT}/../target")
endif()
set(CARGO_TARGET_DIR "${MOLRS_CARGO_TARGET_DEFAULT}" CACHE PATH "Cargo target directory")
get_filename_component(CARGO_TARGET_DIR "${CARGO_TARGET_DIR}" ABSOLUTE BASE_DIR "${MOLRS_ROOT}")
set(MOLRS_TARGET_DIR "${CARGO_TARGET_DIR}/${CARGO_PROFILE}")

# --------------------------------------------------------------------------
# GoogleTest via FetchContent
# --------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --------------------------------------------------------------------------
# Build Rust library (custom target)
# --------------------------------------------------------------------------
# The cdylib, not the staticlib: this test must exercise the same artifact a C
# consumer loads. Suffix/prefix come from CMake because ci-capi runs ubuntu
# (.so), not macOS (.dylib).
set(MOLRS_SHARED_LIB
    "${MOLRS_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}molc${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Let Cargo check source fingerprints on every build; an existing library
# alone does not mean the Rust sources or generated C header are up to date.
add_custom_target(molc_build
    COMMAND cargo build --locked --manifest-path "${MOLRS_CAPI_ROOT}/Cargo.toml"
            --target-dir "${CARGO_TARGET_DIR}" ${MOLRS_CARGO_PROFILE_ARGS}
    BYPRODUCTS "${MOLRS_SHARED_LIB}"
    WORKING_DIRECTORY "${MOLRS_ROOT}"
    COMMENT "Building molc (Rust shared library)"
    VERBATIM
)

# --------------------------------------------------------------------------
# Imported library target
# --------------------------------------------------------------------------
add_library(molc SHARED IMPORTED)
set_target_properties(molc PROPERTIES
    IMPORTED_LOCATION ${MOLRS_SHARED_LIB}
)
# Windows links the import library and loads the DLL at runtime.
if(WIN32)
    if(MSVC)
        set(MOLRS_IMPORT_LIB "${MOLRS_TARGET_DIR}/molc.dll.lib")
    else()
        set(MOLRS_IMPORT_LIB "${MOLRS_TARGET_DIR}/libmolc.dll.a")
    endif()
    set_target_properties(molc PROPERTIES IMPORTED_IMPLIB "${MOLRS_IMPORT_LIB}")
endif()

# No system-library interface list here: that was the staticlib era, where the
# consumer had to re-declare everything Rust's runtime pulls in. A cdylib
# records its own dependencies.

target_include_directories(molc INTERFACE
    ${MOLRS_CAPI_ROOT}/include
)

# --------------------------------------------------------------------------
# Test executable
# --------------------------------------------------------------------------
add_executable(test_molc test_molc.cpp)
add_dependencies(test_molc molc_build)
target_link_libraries(test_molc
    molc
    GTest::gtest_main
)
if(WIN32)
    add_custom_command(TARGET test_molc POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${MOLRS_SHARED_LIB}" "$<TARGET_FILE_DIR:test_molc>"
        VERBATIM
    )
endif()

# The only rpath this binary needs: libmolc's own directory. On ELF its
# SONAME is the bare `libmolc.so`, so the loader has to be told where to
# look. Everything below it is already self-locating: libmolc carries
# molrs inside itself.
set_target_properties(test_molc PROPERTIES
    BUILD_RPATH "${MOLRS_TARGET_DIR}"
    INSTALL_RPATH "${MOLRS_TARGET_DIR}"
)

enable_testing()
add_test(NAME molc_tests COMMAND test_molc)
