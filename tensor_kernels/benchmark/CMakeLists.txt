

find_package(benchmark REQUIRED)

add_executable(kernels_benchmark "main_benchmark.cpp")

target_link_libraries(kernels_benchmark PRIVATE kernels_lib)
target_link_libraries(kernels_benchmark PRIVATE benchmark::benchmark)

target_compile_options(kernels_benchmark PRIVATE -O1)