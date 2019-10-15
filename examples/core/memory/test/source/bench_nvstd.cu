//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "bench_nvstd.cuh"

inline namespace cuda
{
GEANT_DEVICE
int fibonacci(int n)
{
    return (n < 2) ? n : (fibonacci(n-1) + fibonacci(n-2));
}

GEANT_GLOBAL
void loop_update(float* vals, int N)
{
    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < N; i += istride)
    {
        int fib = fibonacci(20);
        float val = N - i + fib;
        atomicAdd(&vals[i], val);
    }
}
}

int compute_grid(int block, int size) { return (size + block - 1) / block; }

void launch_gpu_loop(PTL::TaskGroup<void>& tg, float* vals, int N)
{
    auto func = [](int grid, int block, float* _vals, int _N) {
        //static int counter = 0;
        //auto c = counter++;
        //printf("Launching gpu loop #%i (vals = %p)... grid = %i, block = %i\n", c, (void*) _vals, grid, block);
        TIMEMORY_AUTO_TUPLE(cuda::auto_tuple_t, "");
        loop_update<<<grid, block>>>(_vals, _N);
        auto err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            printf("%s (%i) in %s at line %i\n", cudaGetErrorString(err), err, __FILE__, __LINE__);
        }
        cudaStreamSynchronize(0);
    };
    int block = 32;
    int grid  = compute_grid(block, N);
    tg.run(func, std::move(grid), std::move(block), std::move(vals), std::move(N));
}
