#include "tensor.h"
#include <stdexcept>

float& Tensor::at(uint64_t b, uint64_t i, uint64_t j) {
    if (b >= batch_size || i >= n || j >= m) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data[b * n * m + i * m + j];
}

const float& Tensor::at(uint64_t b, uint64_t i, uint64_t j) const {
    if (b >= batch_size || i >= n || j >= m) {
        throw std::out_of_range("Tensor index out of bounds");
    }
    return data[b * (n * m) + i * m + j];
}

void Tensor::reshape(uint64_t batch, uint64_t height, uint64_t width) {
    if (batch * height * width != batch_size * n * m) {
        throw std::runtime_error("Incompatible Tensor.reshape() input shape");
    }
    batch_size = batch;
    n = height;
    m = width;
};

bool is_close(const Tensor& c1, const Tensor& c2, float epsilon) {
    if (c1.size() != c2.size() ||
        c1.batch_size != c2.batch_size ||
        c1.m != c2.m ||
        c1.n != c2.n) return false;

    // epsilon is absolute tolerancy to float accuracy
    return std::equal(c1.begin(), c1.end(), c2.begin(),
        [epsilon](float a, float b) {
            return std::abs(a - b) < epsilon;
        });
}