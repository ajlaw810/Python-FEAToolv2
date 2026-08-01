cmake_minimum_required(VERSION 3.16)

project(Tet10Checkpoint LANGUAGES CXX)

# ----------------------------------------------------------------------------
# TET10 Checkpoint 1.1 - standalone build
#
# Usage:
#   cmake -B build -DEIGEN3_INCLUDE_DIR=<path-to-eigen>
#   cmake --build build --config Release
#   ./build/tet10_checkpoint        (Linux/macOS)
#   .\build\Release\tet10_checkpoint.exe   (Windows/MSVC)
#
# If Eigen is installed system-wide (e.g. /usr/include/eigen3 on Debian/Ubuntu,
# or via vcpkg with CMAKE_TOOLCHAIN_FILE), find_package(Eigen3) will locate it
# automatically and -DEIGEN3_INCLUDE_DIR is not required.
# ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Locate Eigen3 (header-only) ---------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found via find_package: ${EIGEN3_INCLUDE_DIR}")
elseif(DEFINED EIGEN3_INCLUDE_DIR)
    message(STATUS "Using EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR
        "Eigen3 not found. Pass -DEIGEN3_INCLUDE_DIR=<path-to-eigen> "
        "or install Eigen system-wide.")
endif()

# --- Executable ---------------------------------------------------------------
add_executable(tet10_checkpoint main_tet10_checkpoint.cpp)

target_include_directories(tet10_checkpoint PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<IF:$<BOOL:${Eigen3_FOUND}>,${EIGEN3_INCLUDE_DIR},${EIGEN3_INCLUDE_DIR}>
)

if(MSVC)
    target_compile_options(tet10_checkpoint PRIVATE /O2 /W4)
else()
    target_compile_options(tet10_checkpoint PRIVATE -O2 -Wall -Wextra)
endif()