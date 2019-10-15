
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvfunctional>
#include "timemory/timemory.hpp"
#include "Geant/core/Config.hpp"
#include "PTL/TaskGroup.hh"
#include "PTL/AutoLock.hh"

namespace tim
{
using namespace tim::component;
}

//======================================================================================//

inline namespace cuda
{
using auto_tuple_t =
    tim::auto_tuple<tim::real_clock, tim::thread_cpu_clock, tim::thread_cpu_util>;
GEANT_GLOBAL
void loop_update(float* vals, int N);
}

//======================================================================================//

void launch_gpu_loop(PTL::TaskGroup<void>& tg, float* vals, int N);

//======================================================================================//
