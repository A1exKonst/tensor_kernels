#include <benchmark/benchmark.h>

#include "tensor.h"
#include "matmul.h"
#include "convolution.h"
#include "random_tensor.h"

template <matmul::MatmulType mtype>
static void BM_MatMul(benchmark::State& state) {
    int N = state.range(0);

    Tensor a = generate_tensor(N, 1, N, N);
    Tensor b = generate_tensor(N, 1, N, N);

    for (auto _ : state) {
        Tensor result = matmul::matmul(a, b, mtype);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetComplexityN(N);
    state.counters["Ops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * N * N * N,
        benchmark::Counter::kIsRate
    );
}

template <convolution::ConvType ctype, matmul::MatmulType mtype>
static void BM_Convolution(benchmark::State& state) {
    int N = state.range(0);

    Tensor a = generate_tensor(N, 10, N, N);
    Tensor b = generate_tensor(N, 10, 10, 10);

    for (auto _ : state) {
        //Tensor result = matmul::matmul(a, b, mtype);
        Tensor result = convolution::convolution(a, b, ctype, mtype);

        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    state.SetComplexityN(N);
    state.counters["Ops"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * 2 * N * N * N,
        benchmark::Counter::kIsRate
    );
}
/*
BENCHMARK_TEMPLATE(BM_MatMul, matmul::MatmulType::NAIVE)
->DenseRange(2048,4096,200)
//->Repetitions(5)->DisplayAggregatesOnly(false)
->Unit(benchmark::kMillisecond);
//*/
/*
BENCHMARK_TEMPLATE(BM_MatMul, matmul::MatmulType::CACHE_FRIENDLY)
->DenseRange(512, 2048, 100)
//->Repetitions(5)->DisplayAggregatesOnly(false)
->Unit(benchmark::kMillisecond);
//*/
/*
BENCHMARK_TEMPLATE(BM_MatMul, matmul::MatmulType::TILING)
->DenseRange(10, 512, 10)
//->Repetitions(5)->DisplayAggregatesOnly(false)
->Unit(benchmark::kMillisecond);
//*/
/*
BENCHMARK_TEMPLATE(BM_MatMul, matmul::MatmulType::VECTORIZED)
->DenseRange(10, 512, 10)
//->Repetitions(5)->DisplayAggregatesOnly(false)
->Unit(benchmark::kMillisecond);
//*/
/*
BENCHMARK_TEMPLATE(BM_Convolution, convolution::ConvType::NAIVE, matmul::MatmulType::NAIVE)
->DenseRange(512, 2048, 100)
//->Repetitions(5)->DisplayAggregatesOnly(false)
->Unit(benchmark::kMillisecond);
//*/
BENCHMARK_TEMPLATE(BM_Convolution, convolution::ConvType::IM2COL, matmul::MatmulType::NAIVE)
->DenseRange(10, 512, 10)
//->Repetitions(5)->DisplayAggregatesOnly(false)
->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}

