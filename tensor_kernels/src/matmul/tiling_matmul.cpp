#include "matmul.h"
#include <stdexcept>

namespace matmul {

	Tensor cache_friendly_tiling_matmul(const Tensor& A, const Tensor& B, const size_t block_size) {
		if (A.batch_size != B.batch_size ||
			A.m != B.n) {
			throw std::runtime_error("Tensor matmul: incompatible sizes");
		}

		size_t reduction_dim = A.m;

		Tensor C{ A.batch_size, A.n, B.m };

		for (size_t b = 0; b < C.batch_size; ++b) {

			// Tiling: 
			// i_block -> j_block -> k_block for GPU
			for (size_t i_block = 0; i_block < C.n; i_block += block_size) {
				for (size_t k_block = 0; k_block < reduction_dim; k_block += block_size) {
					for (size_t j_block = 0; j_block < C.m; j_block += block_size) {

						size_t i_end = std::min(i_block + block_size, C.n);
						size_t j_end = std::min(j_block + block_size, C.m);
						size_t k_end = std::min(k_block + block_size, reduction_dim);

						// Cache friendly matmul cycles:
						for (size_t i = i_block; i < i_end; ++i) {
							for (size_t k = k_block; k < k_end; ++k) {
								float a = A(b, i, k);
								for (size_t j = j_block; j < j_end; ++j) {
									C(b, i, j) += a * B(b, k, j);
								}	
							}
						}

					}
				}
			}
		}

		return C;
	};
}