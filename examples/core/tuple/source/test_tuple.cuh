

#include <cstdio>
#if defined(__NVCC__)
#    include <cuda.h>
#    include <cuda_runtime.h>
#    include <device_launch_parameters.h>
#    include <nvfunctional>
#endif

#include "Geant/core/Common.hpp"
#include "Geant/core/Macros.hpp"
#include "Geant/core/Utils.hpp"
#include "PTL/TaskGroup.hh"

template <typename _Func, typename... _Args>
GEANT_HOST_DEVICE void invoker(const _Func& func, _Args&&... args)
{
    func(std::forward<_Args>(args)...);
}

GEANT_DEVICE inline void device_printer() { printf("second\n"); }
inline void              host_printer() { printf("fourth\n"); }

void              launch(TaskGroup<void>&);
GEANT_GLOBAL void kernel();
