
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include <cstdio>
#if defined(__NVCC__)
#    include <cuda.h>
#    include <cuda_runtime.h>
#    include <device_launch_parameters.h>
#    include <nvfunctional>
#endif

#include "Geant/core/Config.hpp"
#include "Geant/core/Utils.hpp"
#include "PTL/TaskGroup.hh"

using namespace PTL;

template <typename _Tp>
using result_of_t = typename std::result_of<_Tp>::type;

template <typename _Tp>
using decay_t = typename std::decay<_Tp>::type;

template <typename _Func, typename... _Args>
GEANT_HOST_DEVICE void invoker(_Func&& func, _Args&&... args)
{
    (nvstd::function<void(_Args...)>(std::forward<_Func>(func),
                                     std::forward<_Args>(args)...))();
}

GEANT_DEVICE inline void device_printer()
{
    printf("[device: %s] second\n", __FUNCTION__);
}
inline void host_printer() { printf("[host: %s] fourth\n", __FUNCTION__); }

void              launch(TaskGroup<void>&);
GEANT_GLOBAL void kernel();

struct Accessor;
struct StepPoint
{
    double pos[3];
    double mom[3];
};

struct Compton
{
    // physical interaction length
    double PIL;
    friend struct Accessor;
};

struct PhotoElectric
{
    // physical interaction length
    double PIL;
    friend struct Accessor;
};

/*
// Ok, I have been able to configure calling generic functions with `nvstd::function`
// on the host and the device and have determined the ideal way to configure the accessors
struct Accessor
{
    GEANT_GLOBAL Accessor(Compton* phys, StepPoint* step)
    {
        phys.PIL = step.pos[threadIdx.x % 2] / step.mom[threadIdx.x % 2];
    }

    GEANT_GLOBAL Accessor(PhotoElectric* obj, StepPoint* step)
    {
        phys.PIL = step.pos[threadIdx.x % 3] / step.mom[threadIdx.x % 3];
    }

    GEANT_GLOBAL Accessor(StepPoint* step, double rand)
    {
        step.pos[0] = 2.0 * (rand - 0.5);
        step.pos[1] = 0.5 * (rand + 1.0);
        step.pos[2] = -0.5 * (rand - 1.0);

        step.mom[0] = step.pos[0] * step.pos[1];
        step.mom[1] = step.pos[1] / step.pos[2];
        step.mom[2] = step.pos[2] + step.pos[0];

        double norm = 0.0;
        for(int i = 0; i < 3; ++i)
            norm += step.mom[i] * step.mom[i];
        for(int i = 0; i < 3; ++i)
            step.mom[i] /= norm;
    }
};

template <typename _Tuple, std::size_t _N = TupleSize<_Tuple>::value>
void test_invoke()
{
    auto func = [&]()
    {
        // where obj_a, obj_b, virt_a, virt_b are all virtual class
        // instances that want to run some calculation on the GPU
        auto device_tuple = MakeTuple(obj_a, obj_b, virt_a, virt_b);
        typedef Tuple<Accessor, Accessor, Accessor, Accessor> ctors_tuple;
        Apply<void>::apply_access<ctors_tuple, decltype(device_tuple)>(object_tuple);
    };

    invoker(func);
}
*/
