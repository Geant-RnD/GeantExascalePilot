
#include "test_tuple.cuh"
#include "test_tuple.hpp"

template <intmax_t N>
void update_array(std::array<std::atomic_intmax_t, N>& vals)
{
    for(intmax_t i = 0; i < N; ++i)
    {
        vals[i] += N - i;
    }
}

template <intmax_t N>
void loop_update_array(const intmax_t& beg, const intmax_t& end,
                       std::array<std::atomic_intmax_t, N>& vals)
{
    printf("[%20s@%i]> beg = %li, end = %li\n", __FUNCTION__, __LINE__, beg, end);
    for(auto i = beg; i < end; ++i)
        update_array<N>(vals);
}

template <intmax_t N>
void print_array(std::array<std::atomic_intmax_t, N>& vals)
{
    std::cout << std::endl;
    for(intmax_t i = 0; i < N; ++i)
        std::cout << "i = " << i << ", value = " << vals[i] << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    write_app_info(argc, argv);

    // create thread-pool with two threads
    intmax_t                            num_threads = GetEnv<intmax_t>("NUM_THREADS", 2);
    ThreadPool                          tp(num_threads);
    static constexpr intmax_t           N = 10;
    std::array<std::atomic_intmax_t, N> vals;
    for(intmax_t i = 0; i < N; ++i)
        vals[i].store(-N + i);

    auto            join = []() { cudaStreamSynchronize(0); };
    TaskGroup<void> tg(join, &tp);

    geantx::cuda::device_query();

    // launches task that runs on GPU
    if(geantx::cuda::device_count() > 0)
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
    tg.run(update_array<N>, std::ref(vals));

    // wait for tasks to finish
    tg.join();
    print_array<N>(vals);

    auto loop_update_args = [&](const intmax_t& beg, const intmax_t& end,
                                std::array<std::atomic_intmax_t, N>& _vals) {
        printf("[%20s@%i]> beg = %li, end = %li\n", __FUNCTION__, __LINE__, beg, end);
        for(auto i = beg; i < end; ++i)
            update_array<N>(_vals);
    };

    auto loop_update_ref = [&](const intmax_t& beg, const intmax_t& end) {
        printf("[%20s@%i]> beg = %li, end = %li\n", __FUNCTION__, __LINE__, beg, end);
        for(auto i = beg; i < end; ++i)
            update_array<N>(vals);
    };

    tg.parallel_for(10, 2, loop_update_args, std::ref(vals));
    tg.parallel_for(10, 3, loop_update_array<N>, std::ref(vals));
    tg.parallel_for(10, 4, loop_update_ref);

    // wait for tasks to finish
    tg.join();

    print_array<N>(vals);

    return EXIT_SUCCESS;
}
