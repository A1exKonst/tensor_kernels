#pragma once
#include <cstdint>
#include <vector>
#include <iostream>

class TensorView;

class Tensor {
private:
    std::vector<float> data;

    friend class TensorView;

public:
    uint64_t batch_size;
    uint64_t n;
    uint64_t m;

    Tensor(const std::vector<float>& data_, 
        uint64_t batch_size_, uint64_t n_, uint64_t m_) : data(data_), 
        batch_size(batch_size_), n(n_), m(m_) {}

    Tensor(uint64_t batch_size_, uint64_t n_, uint64_t m_) :
        batch_size(batch_size_), n(n_), m(m_) {
        data.assign(batch_size * n * m, 0);
    }

    Tensor(std::vector<float>&& data_, 
        uint64_t batch_size_, uint64_t n_, uint64_t m_) : data(std::move(data_)), 
        batch_size(batch_size_), n(n_), m(m_) {}

    Tensor(const Tensor&) = default;

    Tensor(Tensor&&) = default;

    ~Tensor() = default;

    float& at(uint64_t b, uint64_t i, uint64_t j);

    const float& at(uint64_t b, uint64_t i, uint64_t j) const;

    float& operator()(uint64_t b, uint64_t i, uint64_t j) {
        return data[b * (n * m) + i * m + j];
        /* in cycles optimizations of address calculation
           can be missed, that's why implementation is in header file, 
           but operator() is used anyway for OOP purposes.
        */
    };

    const float& operator()(uint64_t b, uint64_t i, uint64_t j) const {
        return data[b * (n * m) + i * m + j];
        /* in cycles optimizations of address calculation
            can be missed, that's why implementation is in header file,
            but operator() is used anyway for OOP purposes.
        */
    };

    void reshape(uint64_t batch, uint64_t height, uint64_t width);

    auto begin() { return data.begin(); };

    auto end() { return data.end(); };

    auto begin() const { return data.begin(); };

    auto end() const { return data.end(); };

    size_t size() const { return data.size(); };
};

bool is_close(const Tensor& c1, const Tensor& c2, float epsilon = 1e-5f);

class TensorView {

};