#include <gtest/gtest.h>

#include "tensor.h"
#include "random_tensor.h"
#include "matmul.h"
#include "convolution.h"

TEST(Matmul, IdentityTensor) {
	Tensor id = Tensor{ {1,0,0,0,1,0,0,0,1}, 1,3,3 };
	Tensor w = Tensor{ {3,0,0,0,5,0,0,0,0},1,3,3 };
	Tensor w_ = matmul::matmul(w, id, matmul::MatmulType::NAIVE);
	EXPECT_TRUE(is_close(w, w_));
}

TEST(Matmul, CacheFriendly) {
	Tensor a = generate_tensor(1, 2, 50, 100);
	Tensor b = generate_tensor(2, 2, 100, 40);
	Tensor naive = matmul::matmul(a, b, matmul::MatmulType::NAIVE);
	Tensor cached = matmul::matmul(a, b, matmul::MatmulType::CACHE_FRIENDLY);

	EXPECT_TRUE(is_close(naive, cached));
}

TEST(Matmul, Tiling) {
	Tensor a = generate_tensor(1, 2, 50, 100);
	Tensor b = generate_tensor(2, 2, 100, 40);
	Tensor naive = matmul::matmul(a, b, matmul::MatmulType::NAIVE);
	Tensor tiling = matmul::matmul(a, b, matmul::MatmulType::TILING);

	EXPECT_TRUE(is_close(naive, tiling));
}

TEST(Matmul, Vectorized) {
	Tensor a = generate_tensor(1, 2, 50, 100);
	Tensor b = generate_tensor(2, 2, 100, 40);
	Tensor naive = matmul::matmul(a, b, matmul::MatmulType::NAIVE);
	Tensor vec = matmul::matmul(a, b, matmul::MatmulType::VECTORIZED);

	EXPECT_TRUE(is_close(naive, vec));
}

TEST(Convolution, Im2Col_1) {
	Tensor w = Tensor{ {3,0,0,0,5,0,0,0,0},1,3,3 };
	Tensor k = Tensor{ {2},1,1,1 };

	Tensor naive_conv = convolution::convolution(w, k, convolution::ConvType::NAIVE);
	Tensor im2col_conv = convolution::convolution(w, k, convolution::ConvType::IM2COL);

	EXPECT_TRUE(is_close(naive_conv, im2col_conv));
}

TEST(Convolution, Im2Col_2) {
	Tensor c = generate_tensor(3, 2, 30, 30);
	Tensor kernel = generate_tensor(4, 2, 5, 5);

	Tensor naive_conv = convolution::convolution(c, kernel, convolution::ConvType::NAIVE);
	Tensor im2col_conv = convolution::convolution(c, kernel, convolution::ConvType::IM2COL, matmul::MatmulType::VECTORIZED);

	EXPECT_TRUE(is_close(naive_conv, im2col_conv));
}