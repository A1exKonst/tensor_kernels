#include "matmul.h"
#include <stdexcept>

namespace matmul {

	Tensor cache_friendly_matmul(const Tensor& A, const Tensor& B) {
		if (A.batch_size != B.batch_size ||
			A.m != B.n) {
			throw std::runtime_error("Tensor matmul: incompatible sizes");
		}

		size_t reduction_dim = A.m;

		Tensor C{ A.batch_size, A.n, B.m };

		for (size_t b = 0; b < C.batch_size; ++b) {

			for (size_t i = 0; i < C.n; ++i) {
				for (size_t k = 0; k < reduction_dim; ++k) {
					const float& a = A(b, i, k);
					for (size_t j = 0; j < C.m; ++j) {
						C(b, i, j) += a * B(b, k, j);

						// Relative to variable "j":
						// A[b,i,k] - is constant
						// B[b,k,j] - is read in lines
						// C[b,i,j] - is written in lines
					}
				}
			}
		}

		return C;
	};
}