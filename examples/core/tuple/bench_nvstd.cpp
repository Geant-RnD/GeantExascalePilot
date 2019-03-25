
#include "test_tuple.cuh"
#include "test_tuple.hpp"

int main(int argc, char** argv)
{
    write_app_info(argc, argv);

    // create thread-pool with two threads
    uintmax_t  num_threads = GetEnv<uintmax_t>("NUM_THREADS", 2);
    ThreadPool tp(num_threads);

    auto            join = []() { cudaStreamSynchronize(0); };
    TaskGroup<void> tg(join, &tp);

    geant::cuda::device_query();

    // launches task that runs on GPU
    if(geant::cuda::device_count() > 0)
    {
        launch(tg);
    }
    // task that runs on CPU
    auto _s = []() { invoker([]() { printf("[host: %s] third\n", __FUNCTION__); }); };
    // task that runs on CPU
    auto _t = []() { invoker(host_printer); };
    // add tasks
    tg.run(_s);
    tg.run(_t);
    // wait for tasks to finish
    tg.join();
}
