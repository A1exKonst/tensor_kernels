#include <stdexcept>

#include "matmul.h"



namespace matmul {

	Tensor matmul(const Tensor& a, const Tensor& b, const MatmulType type) {
		switch (type) {
		case MatmulType::NAIVE:
			return naive_matmul(a, b);
		case MatmulType::CACHE_FRIENDLY:
			return cache_friendly_matmul(a, b);
		case MatmulType::TILING:
			return cache_friendly_tiling_matmul(a, b);
		case MatmulType::VECTORIZED:
			return cache_friendly_tiling_vectorized_matmul(a, b);
		default:
			throw std::out_of_range("No such MatmulType");
		}
	};

}