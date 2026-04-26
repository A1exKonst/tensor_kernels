#include <chrono>

#include "tensor.h"
#include "random_tensor.h"
#include "matmul.h"
#include "convolution.h"


int main() {
	try {
		Tensor a = generate_tensor(1, 2, 50, 100);
		Tensor b = generate_tensor(2, 2, 100, 40);

		std::cout << "Matmul:" << std::endl;

		Tensor naive = matmul::matmul(a, b, matmul::MatmulType::NAIVE);
		Tensor cached = matmul::matmul(a, b, matmul::MatmulType::CACHE_FRIENDLY);
		Tensor tiled = matmul::matmul(a, b, matmul::MatmulType::TILING);
		Tensor vectorized = matmul::matmul(a, b, matmul::MatmulType::VECTORIZED);

		std::cout << "is_close(naive, cached):      " << is_close(naive, cached)		<< std::endl;
		std::cout << "is_close(cached, tiled):      " << is_close(cached, tiled)		<< std::endl;
		std::cout << "is_close(naive, tiled):       " << is_close(naive, tiled)			<< std::endl;
		std::cout << "is_close(naive, vectorized):  " << is_close(naive, vectorized)	<< std::endl;

		Tensor w = Tensor{ {3,0,0,0,5,0,0,0,0},1,3,3 };
		Tensor k = Tensor{ {2},1,1,1 };
		Tensor m = Tensor{ {1,0,0,0,1,0,0,0,1}, 1,3,3 };

		Tensor w_ = matmul::matmul(w, m, matmul::MatmulType::NAIVE);
		std::cout << "is_close(W, E*W) :            " << is_close(w, w_) << std::endl;

		std::cout << std::endl << std::endl << "Convolution:" << std::endl;

		Tensor c = generate_tensor(3, 2, 30, 30);
		Tensor kernel = generate_tensor(4, 2, 5, 5);

		Tensor naive_conv = convolution::convolution(w, k, convolution::ConvType::NAIVE);
		Tensor im2col_conv = convolution::convolution(w, k, convolution::ConvType::IM2COL);
		std::cout << "is_close(naive, im2col):          " << is_close(naive_conv, im2col_conv) << std::endl;

		Tensor naive_conv_2 = convolution::convolution(c, kernel, convolution::ConvType::NAIVE);
		Tensor im2col_conv_2 = convolution::convolution(c, kernel, convolution::ConvType::IM2COL, matmul::MatmulType::VECTORIZED);
		std::cout << "is_close(naive_2, im2col_2):      " << is_close(naive_conv_2, im2col_conv_2) << std::endl;
	}
	catch(std::exception e) {
		std::cout << e.what();
	}

	return 0;
}
