#include <stdexcept>
#include <immintrin.h>

#include "matmul.h"

namespace matmul {

	Tensor cache_friendly_tiling_vectorized_matmul(const Tensor& A, const Tensor& B, const size_t block_size) {
		if (A.batch_size != B.batch_size ||
			A.m != B.n) {
			throw std::runtime_error("Tensor matmul: incompatible sizes");
		}

		size_t reduction_dim = A.m;

		Tensor C{ A.batch_size, A.n, B.m };

		for (size_t b = 0; b < C.batch_size; ++b) {

			// Tiling: 
			for (size_t i_block = 0; i_block < C.n; i_block += block_size) {
				for (size_t j_block = 0; j_block < C.m; j_block += block_size) {
					for (size_t k_block = 0; k_block < reduction_dim; k_block += block_size) {

						// Cache friendly matmul cycles:
						for (size_t i = i_block; i < std::min(i_block + block_size, C.n); ++i) {
							for (size_t k = k_block; k < std::min(k_block + block_size, reduction_dim); ++k) {
								float a = A(b, i, k);
								// A(b,i,k) is constant, duplicated into each element of va
								__m256 va = _mm256_set1_ps(A(b, i, k));

								size_t j_end = std::min(j_block + block_size, C.m);
								size_t j = j_block;

								// vectorized part of matmul:
								for (; j + 7 < j_end; j+=8) {
									__m256 vb = _mm256_loadu_ps(&B(b, k, j));
									__m256 vc = _mm256_loadu_ps(&C(b, i, j));
									vc = _mm256_fmadd_ps(va, vb, vc);
									_mm256_storeu_ps(&C(b, i, j), vc);
									// C(b, i, j) += a * B(b, k, j);
								}

								// scalar tail processing:
								for (; j < j_end; ++j) {
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