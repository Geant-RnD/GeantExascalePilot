
#include "test_tuple.hpp"

//======================================================================================//

typedef Add<double>                           dAdd;
typedef Sub<double>                           dSub;
typedef Mult<double>                          dMult;
typedef Tuple<dAdd, dMult, dSub, dMult, dAdd> HeterogeneousArray;
typedef void (*print_func_t)(double, double, double);

const std::string breaker  = "=============================================";
const std::string notifier = "#" + breaker + " joined tasks " + breaker + "#\n";

//======================================================================================//

int main()
{
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

    // some short hand definitions
    using B       = BaseObject;
    using D       = DerivedObject;
    using AccessB = ObjectAccessor<B>;
    using AccessD = ObjectAccessor<D>;

    // derived object (with access to base class)
    DerivedObject derived_obj;
    // base class object (not derived)
    BaseObject base_obj;

    // create tuple of accessors
    auto access_array = MakeTuple(AccessB(base_obj), AccessD(derived_obj));

    // create tuple of member functions
    auto funct_array = MakeTuple(&AccessB::doSomething, &AccessD::doSomething);

    // apply operator() to all tuple objects (e.g. loop over objects calling operator)
    auto _exec_operator = [&access_array]() { Apply<void>::apply_loop(access_array); };

    // apply doSomething(std::string) to all tuple objects (e.g. loop over objects calling doSomething)
    auto _exec_member_function = [&access_array, &funct_array](const std::string& msg) {
        Apply<void>::apply_functions(access_array, funct_array, msg);
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

    // run the task that calls operator() on each accessor
    tg.run(_exec_operator);
    // run the task that calls doSomething(std::string) on each accessor
    tg.run(_exec_member_function, std::string("member function task worked!"));

    // wait for tasks to finish
    tg.join();
    std::cout << notifier << std::endl;

    tg.run(_exec_operator);

    // wait for tasks to finish
    tg.join();
    std::cout << notifier << std::endl;
}

//======================================================================================//
