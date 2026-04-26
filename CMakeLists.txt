
cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(tensor_kernels)

option(BUILD_TESTING "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

add_subdirectory ("tensor_kernels")

if(BUILD_BENCHMARKS)
    add_subdirectory("tensor_kernels/benchmark")
endif()

if(BUILD_TESTING)
    add_subdirectory("tensor_kernels/tests")
endif()