//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "test_tuple.cuh"

#include "Geant/core/Macros.hpp"

__global__ void kernel()
{
    invoker([] { printf("[kernel: %s] first\n", __FUNCTION__); });
    invoker(device_printer);
}

void launch(TaskGroup<void>& tg)
{
    nvstd::function<void()> func = []() {
        kernel<<<1, 1>>>();
        CUDA_CHECK_LAST_ERROR();
    };
    tg.run(func);
}
