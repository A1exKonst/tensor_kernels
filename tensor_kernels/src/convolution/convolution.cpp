#include <stdexcept>

#include "convolution.h"



namespace convolution {
	Tensor convolution(const Tensor& a, const Tensor& b, ConvType type, matmul::MatmulType mtype) {
		switch (type)
		{
		case convolution::ConvType::NAIVE:
			return naive_convolution(a, b);
			break;
		case convolution::ConvType::IM2COL:
			return im2col_convolution(a, b, mtype);
			break;
		default:
			throw std::out_of_range("No such ConvType found");
			break;
		}
	};
}