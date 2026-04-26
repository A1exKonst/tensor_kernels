
set(LIB_SOURCES
    "src/tensor.cpp"
    "src/random_tensor.cpp"
    
    "src/matmul/matmul.cpp" 
    "src/matmul/naive_matmul.cpp" 
    "src/matmul/cache_matmul.cpp" 
    "src/matmul/tiling_matmul.cpp"
    "src/matmul/vectorized_matmul.cpp" 
    
    "src/convolution/convolution.cpp"
    "src/convolution/naive_convolution.cpp"
    "src/convolution/im2col.cpp"
    "src/convolution/im2col_convolution.cpp"
)

add_library(kernels_lib STATIC ${LIB_SOURCES})

target_include_directories(kernels_lib PUBLIC 
    "${PROJECT_SOURCE_DIR}/tensor_kernels/include"
    "${CMAKE_CURRENT_SOURCE_DIR}" 
    "${CMAKE_BINARY_DIR}"
)

add_executable (tensor_kernels "src/main.cpp")

target_link_libraries(tensor_kernels PRIVATE kernels_lib)


