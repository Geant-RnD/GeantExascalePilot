
#include "test_tuple.cuh"

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
