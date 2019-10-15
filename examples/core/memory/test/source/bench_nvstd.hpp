
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "Geant/core/Config.hpp"
#include "Geant/core/CudaUtils.hpp"
#include "PTL/AutoLock.hh"
#include "PTL/TaskGroup.hh"
#include "timemory/timemory.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <nvfunctional>

namespace tim
{
using namespace tim::component;
}

//======================================================================================//

#define CUDA_ERROR(err) (HandleCudaError(err, __FILE__, __LINE__))

inline void HandleCudaError(cudaError_t err, const char* file, int line)
{
    if(err != cudaSuccess)
    {
        geantx::Log(geantx::kFatal) << cudaGetErrorString(err) << "(" << err << ") in "
                                    << file << " at line " << line;
    }
}

//======================================================================================//
inline namespace cxx
{
using auto_tuple_t =
    tim::auto_tuple<tim::real_clock, tim::thread_cpu_clock, tim::thread_cpu_util>;
void cpu_loop_update(std::vector<std::atomic<float>*>& vals);
}
//======================================================================================//

void launch_cpu_loop(PTL::TaskGroup<void>& tg, std::vector<std::atomic<float>*>& vals);

//======================================================================================//

inline void print_array(float* vals, int N, int sections)
{
    std::cout << std::endl;
    for(int i = 0; i < N; ++i)
    {
        if(i < sections || i > (N - sections))
            std::cout << "i = " << i << ", value = " << std::setw(12)
                      << std::setprecision(2) << std::fixed << vals[i] << std::endl;
        else if(i == sections)
            std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}

//======================================================================================//

inline void print_array(std::vector<std::atomic<float>*>& vals, int sections)
{
    std::cout << std::endl;
    int N = vals.size();
    for(int i = 0; i < N; ++i)
    {
        if(i < sections || i > (N - sections))
            std::cout << "i = " << i << ", value = " << std::setw(12)
                      << std::setprecision(2) << std::fixed << vals[i]->load()
                      << std::endl;
        else if(i == sections)
            std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}

//======================================================================================//
