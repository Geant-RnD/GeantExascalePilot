
#include "test_tuple.cuh"
#include "test_tuple.hpp"

template <uintmax_t N>
void update(uintmax_t vals[N])
{
    for(uintmax_t i = 0; i < N; ++i)
    {
        vals[i] = N - i;
    }
}

int main(int argc, char** argv)
{
    write_app_info(argc, argv);

    // create thread-pool with two threads
    uintmax_t  num_threads = GetEnv<uintmax_t>("NUM_THREADS", 2);
    ThreadPool tp(num_threads);
    static constexpr uintmax_t N = 10;
    uintmax_t vals[N];
    memset(vals, 0, N * sizeof(uintmax_t));

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
    tg.run(update<N>, vals);
    // wait for tasks to finish
    tg.join();

    for(uintmax_t i = 0; i < N; ++i)
        std::cout << "i = " << i << ", value = " << vals[i] << std::endl;
    std::cout << std::endl;
}
