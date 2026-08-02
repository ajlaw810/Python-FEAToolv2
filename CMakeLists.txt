cmake_minimum_required(VERSION 3.16)
project(cpp_python_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

if(NOT EIGEN3_INCLUDE_DIR)
    if(EXISTS "C:/eigen")
        set(EIGEN3_INCLUDE_DIR "C:/eigen")
    else()
        find_package(Eigen3 QUIET)
    endif()
endif()

message(STATUS "Eigen3 include path: ${EIGEN3_INCLUDE_DIR}")

nanobind_add_module(core src/core.cpp)

if(MSVC)
    target_compile_options(core PRIVATE /O2 /fp:fast)
else()
    target_compile_options(core PRIVATE -O3 -ffast-math)
    if(MINGW)
        target_link_options(core PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()

target_include_directories(core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fea
    ${EIGEN3_INCLUDE_DIR}
)

install(TARGETS core DESTINATION cpp_python_project)
