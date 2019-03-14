
#include "test_tuple.hpp"

//======================================================================================//

typedef Add<double>                           dAdd;
typedef Sub<double>                           dSub;
typedef Mult<double>                          dMult;
typedef Tuple<dAdd, dMult, dSub, dMult, dAdd> HeterogeneousArray;
typedef void (*print_func_t)(double, double, double);

//======================================================================================//

int main(int argc, char** argv)
{
    // create thread-pool with two threads
    uintmax_t  num_threads = 2;
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
        Apply<void>::apply_n(ops, start);
        _print_info(start, copy_ops, ops);
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << std::endl;
    };

    auto _exec_object = []() {
        using T = BaseObject;
        using U = DerivedObject;
        // refer to object via derived
        DerivedObject* virt_obj = new DerivedObject();
        // refer to object via base class
        BaseObject* base_obj = new DerivedObject();
        // create tuple
        Tuple<ObjectAccessor<T, U>, ObjectAccessor<T, U>> access_array =
            MakeTuple(ObjectAccessor<T, U>(virt_obj), ObjectAccessor<T, U>(base_obj));
        // apply operator() to all tuple objects (e.g. loop over objects)
        Apply<void>::apply_n(access_array);
        // cleanup
        delete virt_obj;
        delete base_obj;
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

    // run the accessor task
    tg.run(_exec_object);

    // wait for tasks to finish
    tg.join();

    // this failed
    // typedef void (*var_print_t)(dAdd, dSub, dMult, dAdd, dSub);
    // Apply<void>::apply_all<var_print_t>(&variadic_print, ops);
}

//======================================================================================//
