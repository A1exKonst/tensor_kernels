#include <random>
#include <algorithm>
#include <cstdint>

#include "tensor.h"
#include "random_tensor.h"

Tensor generate_tensor(unsigned seed, uint64_t batch_size, uint64_t n, uint64_t m) {
    std::mt19937 gen(seed);

    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // uniform distribution (0.0, 1.0)

    std::vector<float> vec(batch_size * n * m);
    for (float& val : vec) {
        val = dis(gen);
    }

    return Tensor(vec, batch_size, n, m);
}