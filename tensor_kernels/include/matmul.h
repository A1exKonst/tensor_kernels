#pragma once
#include "matmul_type.h"
#include "tensor.h"


namespace matmul {

	Tensor matmul(const Tensor& a, const Tensor& b, const MatmulType type);

	Tensor naive_matmul(const Tensor& a, const Tensor& b);

	Tensor cache_friendly_matmul(const Tensor& a, const Tensor& b);

	Tensor cache_friendly_tiling_matmul(const Tensor& a, const Tensor& b, const size_t block_size = 64);

	Tensor cache_friendly_tiling_vectorized_matmul(const Tensor& a, const Tensor& b, const size_t block_size = 64);
}