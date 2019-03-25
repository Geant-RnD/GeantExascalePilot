

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

template <typename _Tp>
using result_of_t = typename std::result_of<_Tp>::type;

template <typename _Tp>
using decay_t = typename std::decay<_Tp>::type;

template <typename _Func, typename... _Args>
GEANT_HOST_DEVICE void invoker(_Func&& func, _Args&&... args)
{
    (nvstd::function<void(_Args...)>(std::forward<_Func>(func), std::forward<_Args>(args)...))();
    //f();
}

GEANT_DEVICE inline void device_printer() { printf("[device: %s] second\n", __FUNCTION__); }
inline void              host_printer() { printf("[host: %s] fourth\n", __FUNCTION__); }

void              launch(TaskGroup<void>&);
GEANT_GLOBAL void kernel();
