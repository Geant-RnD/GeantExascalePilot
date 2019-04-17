
#pragma once

#if defined(GEANT_USE_TIMEMORY)
#    include "timemory/timemory.hpp"
#else
#    include <Geant/core/Profiler.hpp>
#endif

#include <Geant/core/Tasking.hpp>
#include <Geant/core/Tuple.hpp>
#include <PTL/AutoLock.hh>
#include <PTL/ThreadPool.hh>
#include <PTL/Utility.hh>

#include <atomic>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <type_traits>

typedef std::chrono::duration<double> duration_t;
using Generator = std::mt19937_64;

//======================================================================================//

std::shared_ptr<duration_t>& ref_timer()
{
    typedef std::shared_ptr<duration_t> pointer;
    static pointer                      ptr = pointer(nullptr);
    return ptr;
}

//======================================================================================//

void write_app_info(int argc, char** argv)
{
    std::stringstream ss;
    std::stringstream tmpss;
    tmpss.fill('=');
    tmpss << "\n#" << std::setw(88) << ""
          << "#\n";
    ss << tmpss.str() << "#\n# > Running \"";
    for(int i = 0; i < argc; ++i)
    {
        ss << argv[i] << " ";
    }
    ss << "\"...\n#" << tmpss.str();
    std::cout << ss.str() << std::flush;
}

//======================================================================================//
// macro for reporting the duration between a previous time point and the current time
#if !defined(REPORT_TEST_TIMER)
#    define REPORT_TEST_TIMER(start_time, note, counter, total_count, ref)                                             \
        {                                                                                                              \
            auto       end_time        = std::chrono::high_resolution_clock::now();                                    \
            duration_t elapsed_seconds = end_time - start_time;                                                        \
            if(!ref)                                                                                                   \
            {                                                                                                          \
                ref.reset(new duration_t(elapsed_seconds));                                                            \
            }                                                                                                          \
            auto speed_up = ref->count() / elapsed_seconds.count();                                                    \
            printf("> %-16s :: loop #%lu with %3lu iterations... %10.6f seconds. Speed-up: %5.3f\n", note, counter,    \
                   total_count, elapsed_seconds.count(), speed_up);                                                    \
        }
#endif

//======================================================================================//

template <typename _Tp1, typename _Tp2, typename _Tp3>
void print(_Tp1 val1, _Tp2 val2, _Tp3 val3)
{
    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << "+ values = " << val1 << ", " << val2 << ", " << val3 << "\n" << std::endl;
}

//======================================================================================//

template <typename Arg, typename... Args>
void variadic_print(std::ostream& out, Arg&& arg, Args&&... args)
{
    out << std::forward<Arg>(arg);
    using expander = int[];
    (void) expander{ 0, (void(out << ',' << std::forward<Args>(args)), 0)... };
}

//======================================================================================//
//  this class is used to demonstrate how we can use tuples + template expansion
//  or lambdas to provide an interface that can be overridden, e.g. implementing a
//  custom set of SensitiveDetectors without the need for polymorphism
//
template <typename _Tp>
struct Printer
{
    std::string m_name;
    _Tp         m_value;

    Printer(std::string name, const _Tp& value)
    : m_name(name)
    , m_value(_Tp(value))
    {
    }
    Printer(const Printer& rhs) = default;
    Printer& operator=(const Printer& rhs) = default;
    Printer& operator=(Printer&&) noexcept = default;
    Printer(Printer&& rhs)                 = default;
    ~Printer()                             = default;

    friend std::ostream& operator<<(std::ostream& os, const Printer& obj)
    {
        std::stringstream ss;
        ss.precision(1);
        ss << std::fixed;
        ss << "\t" << std::setw(6) << obj.m_name << " = " << std::setw(6) << (obj.m_value) << std::endl;
        os << ss.str();
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const std::pair<Printer, Printer>& obj)
    {
        std::stringstream ss;
        ss.precision(1);
        ss << std::fixed;
        ss << "  " << std::setw(6) << obj.first.m_name << " : " << std::setw(6) << obj.first.m_value << "   ==> "
           << std::setw(6) << obj.second.m_value << std::endl;
        os << ss.str();
        return os;
    }
};

//======================================================================================//
//  this class is used to demonstrate how we can use tuples + template expansion
//  or lambdas to provide an interface that can be overridden, e.g. implementing a
//  custom set of SensitiveDetectors without the need for polymorphism
//
template <typename _Tp>
struct Add : public Printer<_Tp>
{
    typedef Printer<_Tp> base_type;
    using base_type::m_value;
    Add(const _Tp& val)
    : base_type("add", val)
    {
    }
    // imagine this operator takes a G4Step and records a G4Hit
    void operator()(const _Tp& val) { m_value += val; }
};

//======================================================================================//
//  this class is used to demonstrate how we can use tuples + template expansion
//  or lambdas to provide an interface that can be overridden, e.g. implementing a
//  custom set of SensitiveDetectors without the need for polymorphism
//
template <typename _Tp>
struct Mult : public Printer<_Tp>
{
    typedef Printer<_Tp> base_type;
    using base_type::m_value;
    Mult(const _Tp& val)
    : base_type("mul", val)
    {
    }
    // imagine this operator takes a G4Step and records a G4Hit
    void operator()(const _Tp& val) { m_value *= val; }
};

//======================================================================================//
//  this class is used to demonstrate how we can use tuples + template expansion
//  or lambdas to provide an interface that can be overridden, e.g. implementing a
//  custom set of SensitiveDetectors without the need for polymorphism
//
template <typename _Tp>
struct Sub : public Printer<_Tp>
{
    typedef Printer<_Tp> base_type;
    using base_type::m_value;
    Sub(const _Tp& val)
    : base_type("sub", val)
    {
    }
    // imagine this operator takes a G4Step and records a G4Hit
    void operator()(const _Tp& val) { m_value -= val; }
};

//======================================================================================//

template <typename _Tp, typename... _Args>
class ObjectAccessor;
class ConstructAccessor;
class ConstructAccessorA;
class ConstructAccessorB;

//======================================================================================//
//  this class is sort of dummy demonstration class for the accessor
//
class ObjectA
{
public:
    virtual void operator()()
    {
        std::stringstream ss;
        ss << "+ I am the " << m_class_id << " :: " << __FUNCTION__ << "\n";
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::flush;
    }

    virtual void doSomething(const std::string& msg)
    {
        std::stringstream ss;
        ss << "+ I am the " << m_class_id << " :: " << __FUNCTION__ << ". Message = \"" << msg << "\"\n";
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::flush;
    }

    virtual ~ObjectA() {}

    virtual void generate(Generator& gen)
    {
        int _nquery = (nquery++) % 10;
        if(_nquery == 9)
            m_random_value += 0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

    double GetRandomValue() const { return m_random_value; }
    void   Reset() { m_random_value = 0.0; }

protected:
    double m_random_value = 0.0;

private:
    uintmax_t   nquery     = 0;
    std::string m_class_id = "base class";

private:
    template <typename _Tp, typename... _Args>
    friend class ObjectAccessor;

    friend class ConstructAccessor;
    friend class ConstructAccessorA;
};

//======================================================================================//
//  this class is sort of dummy demonstration class for the accessor
//
class VirtualA : public ObjectA
{
public:
    virtual void generate(Generator& gen)
    {
        int _nquery = (nquery++) % 10;
        if(_nquery == 9)
            m_random_value += -0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

private:
    friend class ConstructAccessor;
    friend class ConstructAccessorA;
    uintmax_t   nquery     = 0;
    std::string m_class_id = "virtual base class";
};

//======================================================================================//
//  this class is sort of dummy demonstration class for the accessor
//
class ObjectB : public ObjectA
{
public:
    virtual void operator()()
    {
        std::stringstream ss;
        ss << "+ I am the " << m_class_id << " :: " << __FUNCTION__ << "\n";
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::flush;
    }

    virtual void doSomething(const std::string& msg)
    {
        std::stringstream ss;
        ss << "+ I am the " << m_class_id << " :: " << __FUNCTION__ << ". Message = \"" << msg << "\"\n";
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::flush;
    }

    virtual void generate(Generator& gen)
    {
        int _nquery = (nquery++) % 10;
        if(_nquery == 0)
            m_random_value += -0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

protected:
    uintmax_t   nquery     = 0;
    std::string m_class_id = "derived class";

private:
    template <typename _Tp, typename... _Args>
    friend class ObjectAccessor;
    friend class ConstructAccessor;
    friend class ConstructAccessorB;
};

//======================================================================================//
//  this class is sort of dummy demonstration class for the accessor
//
class VirtualB : public ObjectB
{
public:
    virtual void generate(Generator& gen)
    {
        int _nquery = (nquery++) % 10;
        if(_nquery == 0)
            m_random_value += 0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

private:
    friend class ConstructAccessor;
    friend class ConstructAccessorB;
    uintmax_t   nquery     = 0;
    std::string m_class_id = "virtual derived class";
};

//======================================================================================//

struct ConstructAccessor
{
    ConstructAccessor(ObjectA* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 9)
            obj->m_random_value += -0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

    ConstructAccessor(VirtualA* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 9)
            obj->m_random_value += -0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

    ConstructAccessor(ObjectB* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 0)
            obj->m_random_value += 0.5 * (std::generate_canonical<double, 24>(gen) - 0.5);
    }

    ConstructAccessor(VirtualB* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 0)
            obj->m_random_value += 0.5 * (std::generate_canonical<double, 24>(gen) - 0.5);
    }
};

struct ConstructAccessorA
{
    ConstructAccessorA(ObjectA* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 9)
            obj->m_random_value += -0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

    ConstructAccessorA(VirtualA* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 9)
            obj->m_random_value += -0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }
};

struct ConstructAccessorB
{
    ConstructAccessorB(ObjectB* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 0)
            obj->m_random_value += 0.5 * (std::generate_canonical<double, 24>(gen) - 0.5);
    }

    ConstructAccessorB(VirtualB* obj, Generator& gen)
    {
        int _nquery = (obj->nquery++) % 10;
        if(_nquery == 0)
            obj->m_random_value += 0.5 * (std::generate_canonical<double, 24>(gen) - 0.5);
    }
};

//======================================================================================//
// this accessor simulates static polymorphism without the explicit need for functional
// polymorphism or static casting. By adding
//
//      template <typename _Tp, typename... _Args>
//      friend class ObjectAccessor;
//
// to the classes to be customized, one can modify the behavior and/or create an
// object that is more compact for data transfers to the GPU
//
template <typename _Tp, typename... _Args>
class ObjectAccessor
{
public:
    typedef std::function<void(_Tp&, _Args...)> Func_t;

public:
    ObjectAccessor(_Tp* obj, const Func_t& func = [](_Tp&, _Args...) {})
    : m_obj(*obj)
    , m_class_id(m_obj.m_class_id)
    , m_random_value(m_obj.m_random_value)
    , m_func(func)
    {
    }

    ObjectAccessor(_Tp& obj, const Func_t& func = [](_Tp&, _Args...) {})
    : m_obj(obj)
    , m_class_id(m_obj.m_class_id)
    , m_random_value(m_obj.m_random_value)
    , m_func(func)
    {
    }

    void operator()(_Args&&... args) { m_func(std::forward<_Tp&>(m_obj), std::forward<_Args>(args)...); }

    // this demonstrates a way to customize
    void doSomething(const std::string& msg)
    {
        m_obj.doSomething(msg);
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << std::endl;
    }

    // when generate(...) is called by ObjectAccessor<ObjectA>, this function is called
    template <typename U = _Tp, std::enable_if_t<(std::is_same<U, ObjectA>::value), int> = 0>
    void generate(Generator& gen)
    {
        int _nquery = (m_obj.nquery++) % 10;
        if(_nquery == 9)
            m_random_value += -0.5 * (std::generate_canonical<double, 12>(gen) - 0.5);
    }

    // when generate(...) is called by ObjectAccessor<ObjectB>, this function is called
    template <typename U = _Tp, std::enable_if_t<(std::is_same<U, ObjectB>::value), int> = 0>
    void generate(Generator& gen)
    {
        int _nquery = (m_obj.nquery++) % 10;
        if(_nquery == 0)
            m_random_value += 0.5 * (std::generate_canonical<double, 24>(gen) - 0.5);
    }

    _Tp&       object() { return m_obj; }
    const _Tp& object() const { return m_obj; }

protected:
    // the base object being accessed
    _Tp& m_obj;
    // reference to a private member of m_obj
    std::string& m_class_id;
    // reference to a protected member of m_obj
    double& m_random_value;
    // a function that can be passed in to customize behavior
    const Func_t& m_func;
};

//======================================================================================//
