#include "convolution.h"
#include "matmul.h"

namespace convolution {
	Tensor im2col_convolution(const Tensor& a, const Tensor& kernel, matmul::MatmulType mtype, const size_t padding, const size_t stride) {

		size_t H_out = (a.n + 2 * padding - kernel.n) / stride + 1;
		size_t W_out = (a.m + 2 * padding - kernel.m) / stride + 1;

		Tensor a_col = im2col(a, kernel.n, kernel.m, padding, stride);

		Tensor kernel_flat = kernel;
		kernel_flat.reshape(1, kernel.batch_size * kernel.m * kernel.n, 1);

		Tensor c = matmul::matmul(a_col, kernel_flat, mtype);

		c.reshape(1, H_out, W_out);

		return c;
	};
}