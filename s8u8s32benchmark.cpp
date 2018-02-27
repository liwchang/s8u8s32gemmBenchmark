//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma warning(push)
#pragma warning(disable:4141 4800)
#include <benchmark/benchmark_api.h>
#pragma warning(pop)

#include "mkl_cblas.h"
#include "windows.h"

using namespace std;

int m_cOutputSize = 9404;
int m_cInputSize = 256;
int m_nPaddedOutputSize = ((m_cOutputSize + 63) & (~63));
int m_nPaddedInputSize = ((m_cInputSize + 63) & (~63));


void MKLML8bit_GEM_BENCHMARK_NOPAD(benchmark::State& state)
{
    int batchSize = state.range(0);
    SetThreadAffinityMask(GetCurrentThread(), 1);

    auto activation = (uint8_t*)_aligned_malloc(m_cInputSize *batchSize * sizeof(uint8_t), 64);
    memset(activation, 0, m_cInputSize *batchSize * sizeof(uint8_t));

    auto weights = (int8_t*)_aligned_malloc(m_cOutputSize * m_cInputSize * sizeof(int8_t), 64);
    memset(weights, 0, m_cOutputSize*m_cInputSize * sizeof(int8_t));

    int32_t* resultMKL;
    resultMKL = (int32_t*)_aligned_malloc(m_cOutputSize *batchSize * sizeof(int32_t), 64);
    memset(resultMKL, 0, m_cOutputSize *batchSize * sizeof(int32_t));

    MKL_INT32 co = 0;

    // warmup
    cblas_gemm_s8u8s32(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_OFFSET::CblasFixOffset, m_cOutputSize, batchSize, m_cInputSize, 1,
        weights, m_cInputSize, 0, activation, m_cInputSize, 0, 0, resultMKL, m_cOutputSize, &co);


    int i = 0;
    for (auto _ : state)
    {
        cblas_gemm_s8u8s32(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_OFFSET::CblasFixOffset, m_cOutputSize, batchSize, m_cInputSize, 1,
            weights, m_cInputSize, 0, activation, m_cInputSize, 0, 0, resultMKL, m_cOutputSize, &co);
        i += (int)(*resultMKL);
    }

    // Make sure the compiler does not optimizes away everything.
    // std::cout << "MKL accumulated value is " << i << std::endl;
}


void MKLML8bit_GEM_BENCHMARK(benchmark::State& state)
{
    int batchSize = state.range(0);
    SetThreadAffinityMask(GetCurrentThread(), 1);

    auto activation = (uint8_t*)_aligned_malloc(m_nPaddedInputSize *batchSize * sizeof(uint8_t), 64);
    memset(activation, 0, m_nPaddedInputSize *batchSize * sizeof(uint8_t));

    auto weights = (int8_t*)_aligned_malloc(m_nPaddedOutputSize * m_nPaddedInputSize * sizeof(int8_t), 64);
    memset(weights, 0, m_nPaddedOutputSize*m_nPaddedInputSize * sizeof(int8_t));

    int32_t* resultMKL;
    resultMKL = (int32_t*)_aligned_malloc(m_nPaddedOutputSize *batchSize * sizeof(int32_t), 64);
    memset(resultMKL, 0, m_nPaddedOutputSize *batchSize * sizeof(int32_t));

    MKL_INT32 co = 0;

    // warmup
    cblas_gemm_s8u8s32(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_OFFSET::CblasFixOffset, m_cOutputSize, batchSize, m_cInputSize, 1,
        weights, m_nPaddedInputSize, 0, activation, m_nPaddedInputSize, 0, 0, resultMKL, m_nPaddedOutputSize, &co);


    int i = 0;
    for (auto _ : state)
    {
        cblas_gemm_s8u8s32(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_OFFSET::CblasFixOffset, m_cOutputSize, batchSize, m_cInputSize, 1,
            weights, m_nPaddedInputSize, 0, activation, m_nPaddedInputSize, 0, 0, resultMKL, m_nPaddedOutputSize, &co);
        i += (int)(*resultMKL);
    }

    // Make sure the compiler does not optimizes away everything.
    // std::cout << "MKL accumulated value is " << i << std::endl;
}

BENCHMARK_CAPTURE(MKLML8bit_GEM_BENCHMARK_NOPAD, Gemm)->Arg(1)->Arg(2)
->Arg(3)->Arg(4)->Arg(5)->Arg(6)
->Arg(7)->Arg(8)->Arg(16)->Arg(32);

BENCHMARK_CAPTURE(MKLML8bit_GEM_BENCHMARK, Gemm)->Arg(1)->Arg(2)
->Arg(3)->Arg(4)->Arg(5)->Arg(6)
->Arg(7)->Arg(8)->Arg(16)->Arg(32);


BENCHMARK_MAIN();