#include "matmul.h"
#include <stdexcept>

namespace matmul {

	Tensor naive_matmul(const Tensor& A, const Tensor& B) {
		if (A.batch_size != B.batch_size ||
			A.m != B.n) {
			throw std::runtime_error("Tensor matmul: incompatible sizes");
		}
		
		size_t reduction_dim = A.m;

		Tensor C{ A.batch_size, A.n, B.m };
		for (size_t b = 0; b < C.batch_size; ++b) {
			for (size_t i = 0; i < C.n; ++i) {
				for (size_t j = 0; j < C.m; ++j) {
					float sum = 0;
					for (size_t k = 0; k < reduction_dim; ++k) {
						sum += A.at(b, i, k) * B.at(b, k, j);
					}
					C(b, i, j) = sum;
				}
			}
		}
		
		return C;
	};
}