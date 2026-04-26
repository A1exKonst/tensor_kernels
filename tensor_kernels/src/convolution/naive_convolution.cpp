#include "convolution.h"

namespace convolution {
	Tensor naive_convolution(const Tensor& A, const Tensor& B) {

		//
		// Main calculation :
		// C[i,j] = sum{k,m,n}(A[k, i+m, j+n] * B[k, m, n])
		// 
		// Limits :
		// k = [0, B.batch_size); 
		// m = [0, B.m); 
		// n = [0, B.n)
		// i = [0, A.m - B.m + 1);
		// j = [0, A.n - B.n + 1);

		if (A.batch_size != B.batch_size ||
			A.n - B.n + 1 <= 0 ||
			A.m - B.m + 1 <= 0) {
			throw std::runtime_error("Incompatible Conv shapes");
		}
		
		Tensor C{ 1, A.n - B.n + 1, A.m - B.m + 1};

		size_t batch_size = A.batch_size;

		// cycles C[i,j]:
		for (size_t i = 0; i < C.n; ++i) {
			for (size_t j = 0; j < C.m; ++j) {
				float sum = 0;
				
				// cycles sum{k,m,n}
				for (size_t k = 0; k < batch_size; ++k) {
					for (size_t n = 0; n < B.n; ++n) {
						for (size_t m = 0; m < B.m; ++m) {
							sum += A.at(k, i + n, j + m) * B.at(k, n, m);
						}
					}
				}

				C.at(0, i, j) = sum;
			}
		}
		return C;
	};
}