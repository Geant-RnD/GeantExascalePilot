
#include "bench_nvstd.hpp"

inline namespace cxx
{
int fibonacci(int n) { return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2)); }

void loop_update(std::vector<std::atomic<float>*>& vals)
{
    for(std::size_t i = 0; i < vals.size(); ++i)
    {
        auto val     = vals.size() - i + fibonacci(20);
        bool success = false;
        do
        {
            float v = vals[i]->load(std::memory_order_relaxed);
            success =
                vals[i]->compare_exchange_strong(v, v + val, std::memory_order_relaxed);
        } while(!success);
    }
}
}

void launch_cpu_loop(PTL::TaskGroup<void>& tg, std::vector<std::atomic<float>*>& vals)
{
    auto func = [](std::vector<std::atomic<float>*>& _vals) {
        // static int counter = 0;
        // auto       c       = counter++;
        // printf("Launching cpu loop #%i (vals = %p)...\n", c, (void*) _vals);
        TIMEMORY_AUTO_TUPLE(cxx::auto_tuple_t, "");
        loop_update(_vals);
    };
    tg.run(func, std::ref(vals));
}
