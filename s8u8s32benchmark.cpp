//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma warning(push)
#pragma warning(disable:4141 4800)
#include "benchmark/benchmark.h"
#pragma warning(pop)
#include <stdlib.h>
#include "mkl_cblas.h"

#ifdef WIN32
#include "windows.h"
#endif

#ifdef _MSC_VER
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

using namespace std;

// if the compiler doesn't have aligned_alloc
#if 1
void* aligned_alloc(size_t alignment, size_t requiredSize)
{
    void* original;  // original pointer
    void** aligned; // aligned pointer
    int offset = alignment - 1 + sizeof(void*);
    if ((original = (void*)malloc(requiredSize + offset)) == NULL)
    {
       return NULL;
    }
    aligned = (void**)(((size_t)(original) + offset) & ~(alignment - 1));
    aligned[-1] = original;
    return aligned;
}
#endif

int m_cOutputSize = 9404;
int m_cInputSize = 256;
int m_nPaddedOutputSize = ((m_cOutputSize + 63) & (~63));
int m_nPaddedInputSize = ((m_cInputSize + 63) & (~63));


void MKLML8bit_GEM_BENCHMARK_NOPAD(benchmark::State& state)
{
    int batchSize = state.range(0);
    
    #ifdef WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
    #endif

    auto activation = (uint8_t*) aligned_alloc(64, m_cInputSize *batchSize * sizeof(uint8_t));
    auto weights = (int8_t*) aligned_alloc(64, m_cOutputSize * m_cInputSize * sizeof(int8_t));
    int32_t* resultMKL;
    resultMKL = (int32_t*) aligned_alloc(64, m_cOutputSize *batchSize * sizeof(int32_t));
    memset(activation, 0, m_cInputSize *batchSize * sizeof(uint8_t));
    memset(weights, 0, m_cOutputSize*m_cInputSize * sizeof(int8_t));
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

    #ifdef WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1);
    #endif

    auto activation = (uint8_t*) aligned_alloc(64, m_nPaddedInputSize *batchSize * sizeof(uint8_t));
    auto weights = (int8_t*) aligned_alloc(64, m_nPaddedOutputSize * m_nPaddedInputSize * sizeof(int8_t));
    int32_t* resultMKL;
    resultMKL = (int32_t*) aligned_alloc(64, m_nPaddedOutputSize *batchSize * sizeof(int32_t));

    memset(activation, 0, m_nPaddedInputSize *batchSize * sizeof(uint8_t));
    memset(weights, 0, m_nPaddedOutputSize*m_nPaddedInputSize * sizeof(int8_t));
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

BENCHMARK(MKLML8bit_GEM_BENCHMARK_NOPAD)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(6)->Arg(7)->Arg(8)->Arg(16)->Arg(32);

BENCHMARK(MKLML8bit_GEM_BENCHMARK)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(6)->Arg(7)->Arg(8)->Arg(16)->Arg(32);

BENCHMARK_MAIN();