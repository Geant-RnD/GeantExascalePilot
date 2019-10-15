//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "Geant/core/CudaDeviceInfo.hpp"
#include "Geant/core/Tasking.hpp"
#include "timemory/timemory.hpp"

#include "bench_nvstd.cuh"
#include "bench_nvstd.hpp"

using namespace PTL;
using namespace tim::component;

int main(int argc, char** argv)
{
    // write_app_info(argc, argv);
    tim::timemory_init(argc, argv);

    // create thread-pool with two threads
    int        num_threads = geantx::GetEnv<int>("NUM_THREADS", 4);
    ThreadPool gpu_tp(num_threads);
    ThreadPool cpu_tp(num_threads);

    static constexpr int             N        = 1024;
    int                              nloop    = 1000;
    float*                           cpu_vals = (float*) malloc(N * sizeof(float));
    float*                           gpu_vals = nullptr;
    std::vector<std::atomic<float>*> flt_vals(N, nullptr);
    for(auto& itr : flt_vals)
        itr = new std::atomic<float>(0.0f);

    // geantx::cudaruntime::DeviceQuery();
    cudaSetDevice(0);

    CUDA_ERROR(cudaMalloc(&gpu_vals, N * sizeof(float)));
    CUDA_ERROR(cudaMemset(gpu_vals, 0, N * sizeof(float)));
    memset(cpu_vals, 0, N * sizeof(float));

    using cpu_tuple_t = tim::component_tuple<real_clock, cpu_clock, cpu_util>;
    using gpu_tuple_t = tim::component_tuple<real_clock, cpu_clock, cpu_util, cuda_event>;
    gpu_tuple_t tgpu("GPU timer");
    cpu_tuple_t tcpu("CPU timer");

    tgpu.start();
    tcpu.start();
    auto gpu_join = [&]() {
        tgpu.stop();
        printf("[gpu] joined...\n");
        cudaStreamSynchronize(0);
    };
    auto cpu_join = [&]() {
        tcpu.stop();
        printf("[cpu] joined...\n");
    };

    TaskGroup<void> gpu_tg(gpu_join, &gpu_tp);
    TaskGroup<void> cpu_tg(cpu_join, &cpu_tp);

    {
        TIMEMORY_AUTO_TUPLE(cxx::auto_tuple_t, "");
        for(int i = 0; i < nloop; ++i)
        {
            launch_cpu_loop(cpu_tg, flt_vals);
            launch_gpu_loop(gpu_tg, gpu_vals, N);
        }
        // wait for tasks to finish
        gpu_tg.join();
        cpu_tg.join();
    }

    std::cout << tgpu << std::endl;
    std::cout << tcpu << std::endl;

    print_array(flt_vals, 10);
    CUDA_ERROR(cudaMemcpy(cpu_vals, gpu_vals, N * sizeof(float), cudaMemcpyDeviceToHost));
    print_array(cpu_vals, N, 10);

    delete[] cpu_vals;
    cudaFree(gpu_vals);
    for(auto& itr : flt_vals)
        delete itr;

    return EXIT_SUCCESS;
}
