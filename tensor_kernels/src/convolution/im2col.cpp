#include "convolution.h"

namespace convolution {
	Tensor im2col(const Tensor& A, const size_t H_k, const size_t W_k, const size_t padding, const size_t stride) {

		// 
		// A.shape = (B,H,W)
		// H_out = floor((H + 2P - H_k) / S) + 1
		// W_out = floor((W + 2P - W_k) / S) + 1
		// V.shape = (B*H_out*W_out, H_k*W_k)
		// 
		// V[i,j] = A[b, y + dy, x + dx]
		// i = b*(H_out*W_out) + y_out*W_out + x_out
		// j = dy * W_k + dx
		// y = y_out * S
		// x = x_out * S
		// 
		// S - stride
		// 
		// b = [0, batch_size)
		// y_out = [0, H_out - 1)
		// x_out = [0, W_out - 1)
		// dy = [0, H_k - 1)
		// dx = [0, W_k - 1)
		//

		size_t H_out = (A.n + 2 * padding - H_k) / stride + 1;
		size_t W_out = (A.m + 2 * padding - W_k) / stride + 1;

		Tensor C{ 1, H_out * W_out, A.batch_size * H_k * W_k };

		
		for (size_t y_out = 0; y_out < H_out; ++y_out) {
			for (size_t x_out = 0; x_out < W_out; ++x_out) {

				size_t i = y_out * W_out + x_out;
				size_t y = y_out * stride;
				size_t x = x_out * stride;
				for (size_t b = 0; b < A.batch_size; ++b) {
					size_t channel_offset = b * H_k * W_k;
					for (size_t dy = 0; dy < H_k; ++dy) {
						for (size_t dx = 0; dx < W_k; ++dx) {
							// j = dy * W_k + dx
							// V[i,j] = A[b, y + dy, x + dx]
							// A and V are iterated linearly through dx
							size_t j = channel_offset + dy*W_k + dx;
							C(0, i, j) = A(b, y + dy, x + dx);
						}
					}
				}
			}
		}
		/*
		for (size_t b = 0; b < A.batch_size; ++b) {
			for (size_t y_out = 0; y_out < H_out; ++y_out) {
				for (size_t x_out = 0; x_out < W_out; ++x_out) {
					// i = b*(H_out*W_out) + y_out*W_out + x_out
					// y = y_out * S
					// x = x_out * S
					size_t i = b * H_out * W_out + y_out * W_out + x_out;
					size_t y = y_out * stride;
					size_t x = x_out * stride;
					for (size_t dy = 0; dy < H_k; ++dy) {
						size_t v_offset = dy * W_k;
						size_t y_a = y + dy;
						for (size_t dx = 0; dx < W_k; ++dx) {
							// j = dy * W_k + dx
							// V[i,j] = A[b, y + dy, x + dx]
							// A and V are iterated linearly through dx
							size_t j = v_offset + dx;
							C(0, i, j) = A(b, y_a, x + dx);
						}
					}
				}
			}
		}
		*/
		return C;
	};
}