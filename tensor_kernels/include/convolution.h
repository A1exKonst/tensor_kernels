#pragma once
#include "tensor.h"
#include "matmul.h"
#include "convolution_type.h"

namespace convolution {
	Tensor convolution(const Tensor& a, const Tensor& b, ConvType type, matmul::MatmulType mtype = matmul::MatmulType::NAIVE);

	Tensor naive_convolution(const Tensor& a, const Tensor& b);

	Tensor im2col_convolution(const Tensor& a, const Tensor& b, matmul::MatmulType mtype, const size_t padding = 0, const size_t stride = 1);

	Tensor im2col(const Tensor& A, const size_t H_k, const size_t W_k, const size_t padding = 0, const size_t stride = 1);
}