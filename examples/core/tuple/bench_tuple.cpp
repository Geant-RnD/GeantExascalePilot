
#include "PTL/Utility.hh"
#include "test_tuple.hpp"

typedef std::chrono::duration<double> duration_t;

//======================================================================================//
// macro for recording a time point
#if !defined(GET_TIMER)
#    define GET_TIMER(var) auto var = std::chrono::high_resolution_clock::now()
#endif

//======================================================================================//
// macro for reporting the duration between a previous time point and the current time
#if !defined(REPORT_TIMER)
#    define REPORT_TIMER(start_time, note, counter, total_count, ref)                                                  \
        {                                                                                                              \
            auto       end_time        = std::chrono::high_resolution_clock::now();                                    \
            duration_t elapsed_seconds = end_time - start_time;                                                        \
            if(!ref)                                                                                                   \
            {                                                                                                          \
                ref.reset(new duration_t(elapsed_seconds));                                                            \
            }                                                                                                          \
            auto speed_up = ref->count() / elapsed_seconds.count();                                                    \
            printf("> %-20s :: loop #%lu with %3lu iterations... %10.6f seconds. Speed-up: %5.3f\n", note, counter,    \
                   total_count, elapsed_seconds.count(), speed_up);                                                    \
        }
#endif

//======================================================================================//

// some short hand definitions
using B         = BaseObject;
using D         = DerivedObject;
using AccessB   = ObjectAccessor<B>;
using AccessD   = ObjectAccessor<D>;
using Generator = std::mt19937_64;

//======================================================================================//

template <typename _Func, typename... _Args>
void exec(_Func&& _func, _Args&&... _args)
{
    _func(std::forward<_Args>(_args)...);
}

//======================================================================================//

void run(uintmax_t nloop, uintmax_t nitr)
{
    // create thread-pool with two threads
    // uintmax_t       num_threads = GetEnv<uintmax_t>("NUM_THREADS", 2);
    // ThreadPool      tp(num_threads);
    // TaskGroup<void> tg(&tp);
    // constexpr std::size_t unroll_length = 10;
    // uintmax_t _nitr = nitr / unroll_length;
    // uintmax_t _nmod = nloop % nitr;

    // reference timer
    std::random_device                     rd;
    std::shared_ptr<duration_t>            ref_timer;
    auto                                   seed = rd();
    std::mt19937_64                        f_generator(seed);
    std::mt19937_64                        t_generator(seed);
    std::mt19937_64                        l_generator(seed);
    std::vector<std::pair<double, double>> results;

    // base class object (not derived)
    BaseObject base_obj;
    // derived object (with access to base class)
    DerivedObject derived_obj;

    auto record = [&]() {
        results.push_back(std::make_pair(base_obj.GetRandomValue(), derived_obj.GetRandomValue()));
        base_obj.Reset();
        derived_obj.Reset();
    };

    //==================================================================================//
    //              Functional polymorphism section
    //==================================================================================//
    // functional operators
    auto                               b_funct      = [&]() { base_obj.generate(f_generator); };
    auto                               d_funct      = [&]() { derived_obj.generate(f_generator); };
    std::vector<std::function<void()>> funct_array  = { b_funct, d_funct };
    auto                               funct_vector = [&]() {
        for(const auto& itr : funct_array)
            itr();
    };

    GET_TIMER(funct_1);
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_vector();
    REPORT_TIMER(funct_1, "    Functional polymorphism", nloop, nitr, ref_timer);
    record();

    //==================================================================================//
    //          Tuple Accessor section
    //==================================================================================//
    // create tuple of member functions
    auto b_tuple     = [&]() { base_obj.generate(t_generator); };
    auto d_tuple     = [&]() { derived_obj.generate(t_generator); };
    auto funct_tuple = [&]() { Apply<void>::apply_loop(MakeTuple(b_tuple, d_tuple)); };

    GET_TIMER(tuple_1);
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_tuple();
    REPORT_TIMER(tuple_1, "Accessor Tuple polymorphism", nloop, nitr, ref_timer);
    record();

    //==================================================================================//
    //          Lambda section
    //==================================================================================//
    // create tuple of member functions
    auto b_lambda    = [&]() { base_obj.generate(l_generator); };
    auto d_lambda    = [&]() { derived_obj.generate(l_generator); };
    auto funct_lamda = [&]() {
        b_lambda();
        d_lambda();
    };

    GET_TIMER(tuple_2);
    // run the task that calls generate(...) on each tuple accessor
    for(uintmax_t i = 0; i < nitr; ++i)
        funct_lamda();
    REPORT_TIMER(tuple_2, "        Lambda polymorphism", nloop, nitr, ref_timer);
    record();

    std::cout << "\n+ Random seed = " << seed << std::endl;
    for(auto& itr : results)
        std::cout << "      " << std::fixed << std::setw(8) << std::setprecision(2) << itr.first << ", " << std::setw(8)
                  << std::setprecision(2) << itr.second << std::endl;
}

//======================================================================================//

int main(int argc, char** argv)
{
    uintmax_t nloop = 5;
    uintmax_t nitr  = 10000000;

    if(argc > 1)
        nloop = static_cast<uintmax_t>(atol(argv[1]));
    if(argc > 2)
        nitr = static_cast<uintmax_t>(atol(argv[2]));

    for(uintmax_t i = 0; i < nloop; ++i)
    {
        std::cerr << std::endl;
        run(i + 1, (i + 1) * nitr);
    }
    std::cerr << std::endl;
}

//======================================================================================//
