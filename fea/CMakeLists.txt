cmake_minimum_required(VERSION 3.16)

project(Tet10Checkpoint LANGUAGES CXX)

# ----------------------------------------------------------------------------
# TET10 Checkpoints - standalone builds
#
# Usage:
#   cmake -B build -DEIGEN3_INCLUDE_DIR=<path-to-eigen>
#   cmake --build build --config Release
#   ./build/tet10_checkpoint        (Linux/macOS)
#   .\build\Release\tet10_checkpoint.exe   (Windows/MSVC)
#
# Executables:
#   tet10_checkpoint   - Checkpoint 1.1: TET10 stiffness + eigenvalue spectrum
#   beam_checkpoint    - Checkpoint 1.2: global sparse assembly + cantilever
#                        beam theoretical benchmark
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

# --- Shared target configuration ----------------------------------------------
# Adds a checkpoint executable with the common compile flags and Eigen include
# path (plus the fea/ source dir for the local headers).
function(add_checkpoint_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /O2 /W4)
    else()
        target_compile_options(${target} PRIVATE -O2 -Wall -Wextra)
    endif()
endfunction()

# --- Checkpoint 1.1 ------------------------------------------------------------
add_checkpoint_executable(tet10_checkpoint main_tet10_checkpoint.cpp)

# --- Checkpoint 1.2 ------------------------------------------------------------
# Global sparse assembly + cantilever beam theoretical benchmark.
add_checkpoint_executable(beam_checkpoint main_beam_checkpoint.cpp)