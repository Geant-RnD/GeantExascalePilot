
#pragma once

#include <Geant/core/Tasking.hpp>
#include <Geant/core/Tuple.hpp>
#include <PTL/AutoLock.hh>
#include <PTL/ThreadPool.hh>
#include <iomanip>

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
    /*
    : m_name(rhs.m_name)
    , m_value(new _Tp(*rhs.m_value))
    {
    }*/
    Printer& operator=(const Printer& rhs) = default;
    /*
    {
        if(this != &rhs)
        {
            m_name   = rhs.m_name;
            *m_value = *rhs.m_value;
        }
        return *this;
    }*/
    Printer& operator=(Printer&&) noexcept = default;
    Printer(Printer&& rhs)                 = default;
    ~Printer()
    {
        // delete m_value;
    }

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

template <typename _Tp>
struct Add : public Printer<_Tp>
{
    typedef Printer<_Tp> base_type;
    using base_type::m_value;
    Add(const _Tp& val)
    : base_type("add", val)
    {
    }
    void operator()(const _Tp& val) { m_value += val; }
};

//======================================================================================//

template <typename _Tp>
struct Mult : public Printer<_Tp>
{
    typedef Printer<_Tp> base_type;
    using base_type::m_value;
    Mult(const _Tp& val)
    : base_type("mul", val)
    {
    }
    void operator()(const _Tp& val) { m_value *= val; }
};

//======================================================================================//

template <typename _Tp>
struct Sub : public Printer<_Tp>
{
    typedef Printer<_Tp> base_type;
    using base_type::m_value;
    Sub(const _Tp& val)
    : base_type("sub", val)
    {
    }
    void operator()(const _Tp& val) { m_value -= val; }
};

//======================================================================================//

typedef Add<double>                           dAdd;
typedef Sub<double>                           dSub;
typedef Mult<double>                          dMult;
typedef Tuple<dAdd, dMult, dSub, dMult, dAdd> HeterogeneousArray;
typedef void (*print_func_t)(double, double, double);

//======================================================================================//
