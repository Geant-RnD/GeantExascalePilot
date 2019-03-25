
#include "test_tuple.hpp"

//======================================================================================//

typedef Add<double>                           dAdd;
typedef Sub<double>                           dSub;
typedef Mult<double>                          dMult;
typedef Tuple<dAdd, dMult, dSub, dMult, dAdd> HeterogeneousArray;
typedef void (*print_func_t)(double, double, double);

const std::string breaker  = "-------------------------------------";
const std::string notifier = "#" + breaker + " joined tasks " + breaker + "#\n";

//======================================================================================//

// some short hand definitions
using Generator = std::mt19937_64;
using A         = ObjectA;
using B         = ObjectB;
using AccessA   = ObjectAccessor<A, Generator&>;
using AccessB   = ObjectAccessor<B, Generator&>;
using Aaccess   = ObjectAccessor<A>;
using Baccess   = ObjectAccessor<B>;

//======================================================================================//

int main(int argc, char** argv)
{
    write_app_info(argc, argv);

    // reference timer
    std::random_device                     rd;
    std::shared_ptr<duration_t>            ref_timer;
    auto                                   seed = rd();
    Generator                              r_generator(seed);
    Generator                              f_generator(seed);
    std::vector<std::pair<double, double>> results;

    auto record = [&](ObjectA* obj_a, ObjectB* obj_b) {
        results.push_back(std::make_pair(obj_a->GetRandomValue(), obj_b->GetRandomValue()));
        obj_a->Reset();
        obj_b->Reset();
        r_generator = Generator(seed);
        f_generator = Generator(seed);
    };
    std::cout << "\n+ Random seed = " << seed << std::endl;
    auto report = [&]() {
        for(auto& itr : results)
        {
            std::cout << "      " << std::fixed << std::setw(8) << std::setprecision(2) << itr.first << ", "
                      << std::setw(8) << std::setprecision(2) << itr.second << std::endl;
        }
        std::cout << std::endl;
    };

    // create thread-pool with two threads
    uintmax_t  num_threads = GetEnv<uintmax_t>("NUM_THREADS", 2);
    ThreadPool tp(num_threads);

    // here I make a tuple of doubles
    Tuple<double, double, double> three_vec = MakeTuple(1.0, 2.0, 3.0);
    // here I make a tuple of structs that are NOT polymorphic
    HeterogeneousArray _ops = MakeTuple(dAdd(2.0), dMult(1.0), dSub(0.0), dMult(-2.0), dAdd(-1.0));

    // here I just demonstrate expanding a tuple into arguments, e.g.
    // Tuple<double, double, double> expands to print(double, double, double)
    auto _print = [&]() { Apply<void>::apply_all<print_func_t>(print, three_vec); };

    // this prints out the tuple of heterogeneous structs
    auto _print_info = [](const double& start, const HeterogeneousArray& lhs, const HeterogeneousArray& rhs) {
        std::stringstream ss;
        ss << std::right << std::setw(16) << "+ result of applying " << std::setw(4) << std::setprecision(1)
           << std::fixed << start << ":\n"
           << std::endl
           << std::make_pair(Get<0>(lhs), Get<0>(rhs)) << std::make_pair(Get<1>(lhs), Get<1>(rhs))
           << std::make_pair(Get<2>(lhs), Get<2>(rhs)) << std::make_pair(Get<3>(lhs), Get<3>(rhs))
           << std::make_pair(Get<4>(lhs), Get<4>(rhs));
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::flush;
    };

    // this executes the heterogenous structs
    auto _exec_hetero = [_print_info](const double& start, HeterogeneousArray ops) {
        auto copy_ops = ops;
        Apply<void>::apply_loop(ops, start);
        _print_info(start, copy_ops, ops);
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << std::endl;
    };

    // derived object (with access to base class)
    ObjectB obj_b;
    // base class object (not derived)
    ObjectA obj_a;

    std::function<void(ObjectA&, Generator&)> op_a = [&](ObjectA& m_obj, Generator& gen) { m_obj.generate(gen); };
    std::function<void(ObjectB&, Generator&)> op_b = [&](ObjectB& m_obj, Generator& gen) { m_obj.generate(gen); };

    auto access_a = AccessA(obj_a, op_a);
    auto access_b = AccessB(obj_b, op_b);
    auto a_access = Aaccess(obj_a);
    auto b_access = Baccess(obj_b);

    // create tuple of accessors
    auto access_array = MakeTuple(access_a, access_b);
    // create tuple of member functions
    auto funct_array = MakeTuple(&AccessA::doSomething, &AccessB::doSomething);

    // apply operator() to all tuple objects (e.g. loop over objects calling operator)
    auto _exec_operator = [&](Generator& gen) {
        for(uint32_t i = 0; i < 50; ++i)
        {
            access_a(std::ref(gen));
            access_b(std::ref(gen));
        }
        record(&(access_a.object()), &(access_b.object()));
        for(uint32_t i = 0; i < 50; ++i)
        {
            // defined in source/Geant/core/Tuple.hpp (towards end of file)
            Apply<void>::apply_loop(MakeTuple(access_a, access_b), std::ref(gen));
        }
        record(&(access_a.object()), &(access_b.object()));
    };

    // apply doSomething(std::string) to all tuple objects (e.g. loop over objects calling doSomething)
    auto _exec_member_function = [&](const std::string& msg) {
        // defined in source/Geant/core/Tuple.hpp (towards end of file)
        Apply<void>::apply_functions(access_array, funct_array, msg);
    };

    auto _exec_custom_operator = [&]() {
        a_access();
        b_access();
    };
    // create task-group that uses thread-pool
    TaskGroup<void> tg(&tp);

    // add tasks
    tg.run(_exec_hetero, -2.0, _ops);
    tg.run(_exec_hetero, -1.0, _ops);
    tg.run(_exec_hetero, 0.0, _ops);
    tg.run(_exec_hetero, 1.0, _ops);
    tg.run(_exec_hetero, -2.0, _ops);
    tg.run(_print);
    tg.run(_print);
    tg.run(_print);
    tg.run(_print);

    // wait for tasks to finish
    tg.join();
    std::cout << notifier << std::endl;

    // run the task that calls doSomething(std::string) on each accessor
    tg.run(_exec_member_function, std::string("member function task worked!"));
    tg.run(_exec_custom_operator);

    // wait for tasks to finish
    tg.join();

    // run the task that calls operator() on each accessor
    tg.run(_exec_operator, std::ref(r_generator));

    // wait for tasks to finish
    tg.join();
    std::cout << notifier << std::endl;
    report();

    // add a task
    tg.run(_exec_operator, std::ref(r_generator));

    // wait for tasks to finish
    tg.join();
    std::cout << notifier << std::endl;
    report();
}

//======================================================================================//
