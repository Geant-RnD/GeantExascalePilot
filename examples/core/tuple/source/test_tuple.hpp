
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

template <typename _Tp>
class ObjectAccessor;

//======================================================================================//

class BaseObject
{
public:
    virtual void operator()()
    {
        std::stringstream ss;
        ss << "+ I am the " << m_class_id << "\n";
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::flush;
    }

    virtual ~BaseObject() {}

private:
    std::string m_class_id = "base class";

private:
    template <typename _Tp>
    friend class ObjectAccessor;
};

//======================================================================================//

class DerivedObject : public BaseObject
{
public:
    virtual void operator()()
    {
        std::stringstream ss;
        ss << "+ I am the " << m_class_id << "\n";
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << ss.str() << std::flush;
    }

private:
    std::string m_class_id = "derived class";

private:
    template <typename _Tp>
    friend class ObjectAccessor;
};

//======================================================================================//

template <typename _Tp>
class ObjectAccessor
{
public:
    ObjectAccessor(_Tp* obj)
    : m_obj(obj)
    {
    }

    void operator()()
    {
        m_obj->operator()();
        m_obj->m_class_id += " (modified via accessor)";
        m_obj->operator()();

        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << std::endl;
    }

private:
    _Tp* m_obj;
};

//======================================================================================//
