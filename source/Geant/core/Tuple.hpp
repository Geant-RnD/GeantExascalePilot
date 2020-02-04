//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Reimplementation of std::tuple for use in device code.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

//======================================================================================//

template <std::size_t...>
struct _IndexTuple;

template <typename...>
using _Void_t = void;

// Stores a tuple of indices.  Used by tuple and pair, and by bind() to
// extract the elements in a tuple.
template <size_t... _Indexes>
struct _IndexTuple {};

// Concatenates two _IndexTuples.
template <typename _Itup1, typename _Itup2>
struct ITupleCat;

template <size_t... _Ind1, size_t... _Ind2>
struct ITupleCat<_IndexTuple<_Ind1...>, _IndexTuple<_Ind2...>> {
  using __type = _IndexTuple<_Ind1..., (_Ind2 + sizeof...(_Ind1))...>;
};

// Builds an _IndexTuple<0, 1, 2, ..., _Num-1>.
template <size_t _Num>
struct _BuildIndexTuple : ITupleCat<typename _BuildIndexTuple<_Num / 2>::__type,
                                    typename _BuildIndexTuple<_Num - _Num / 2>::__type> {
};

template <>
struct _BuildIndexTuple<1> {
  typedef _IndexTuple<0> __type;
};

template <>
struct _BuildIndexTuple<0> {
  typedef _IndexTuple<> __type;
};

/// Finds the size of a given tuple type.
template <typename _Tp>
struct TupleSize;

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// 2770. TupleSize<const T> specialization is not SFINAE compatible

#if __cplusplus <= 201402L
template <typename _Tp, typename = void>
struct __TupleSize_cv_impl {};

template <typename _Tp>
struct __TupleSize_cv_impl<_Tp, _Void_t<decltype(TupleSize<_Tp>::value)>>
    : std::integral_constant<size_t, TupleSize<_Tp>::value> {};

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// 2313. TupleSize should always derive from integral_constant<size_t, N>
template <typename _Tp>
struct TupleSize<const _Tp> : __TupleSize_cv_impl<_Tp> {};

template <typename _Tp>
struct TupleSize<volatile _Tp> : __TupleSize_cv_impl<_Tp> {};

template <typename _Tp>
struct TupleSize<const volatile _Tp> : __TupleSize_cv_impl<_Tp> {};
#else
template <typename _Tp, typename _Up = typename remove_cv<_Tp>::type,
          typename = typename enable_if<is_same<_Tp, _Up>::value>::type,
          size_t   = TupleSize<_Tp>::value>
using __enable_if_has_TupleSize = _Tp;

template <typename _Tp>
struct TupleSize<const __enable_if_has_TupleSize<_Tp>> : public TupleSize<_Tp> {};

template <typename _Tp>
struct TupleSize<volatile __enable_if_has_TupleSize<_Tp>> : public TupleSize<_Tp> {};

template <typename _Tp>
struct TupleSize<const volatile __enable_if_has_TupleSize<_Tp>> : public TupleSize<_Tp> {
};
#endif

/// Gives the type of the ith element of a given tuple type.
template <std::size_t __i, typename _Tp>
struct TupleElement;

// Duplicate of C++14's TupleElement_t for internal use in C++11 mode
template <std::size_t __i, typename _Tp>
using __TupleElement_t = typename TupleElement<__i, _Tp>::type;

template <std::size_t __i, typename _Tp>
struct TupleElement<__i, const _Tp> {
  typedef typename std::add_const<__TupleElement_t<__i, _Tp>>::type type;
};

template <std::size_t __i, typename _Tp>
struct TupleElement<__i, volatile _Tp> {
  typedef typename std::add_volatile<__TupleElement_t<__i, _Tp>>::type type;
};

template <std::size_t __i, typename _Tp>
struct TupleElement<__i, const volatile _Tp> {
  typedef typename std::add_cv<__TupleElement_t<__i, _Tp>>::type type;
};

#if __cplusplus > 201103L
#  define __cpp_lib_TupleElement_t 201402

template <std::size_t __i, typename _Tp>
using TupleElement_t = typename TupleElement<__i, _Tp>::type;
#endif

//======================================================================================//

template <typename... _Elements>
class Tuple;

template <typename _Tp>
struct _IsEmptyNonTuple : std::is_empty<_Tp> {};

// Using EBO for elements that are tuples causes ambiguous base errors.
template <typename _El0, typename... _El>
struct _IsEmptyNonTuple<Tuple<_El0, _El...>> : std::false_type {};

// Use the Empty Base-class Optimization for empty, non-final types.
// template <typename _Tp>
// using _EmptyNotFinal = typename std::conditional<std::is_final<_Tp>, std::false_type,
// _IsEmptyNonTuple<_Tp>>::type; Use the Empty Base-class Optimization for empty,
// non-final types.
template <typename _Tp>
using _EmptyNotFinal =
    typename std::conditional<std::is_final<_Tp>::value, std::false_type,
                              _IsEmptyNonTuple<_Tp>>::type;

template <std::size_t _Idx, typename _Head, bool = _EmptyNotFinal<_Head>::value>
struct _HeadBase;

template <typename _Tp>
class _ReferenceWrapper;

// Helper which adds a reference to a type when given a _ReferenceWrapper
template <typename _Tp>
struct _StripReferenceWrapper {
  typedef _Tp __type;
};

template <typename _Tp>
struct _StripReferenceWrapper<_ReferenceWrapper<_Tp>> {
  typedef _Tp &__type;
};

template <typename _Tp>
struct _DecayAndStrip {
  typedef typename _StripReferenceWrapper<typename std::decay<_Tp>::type>::__type __type;
};

template <std::size_t _Idx, typename _Head>
struct _HeadBase<_Idx, _Head, true> : public _Head {
  constexpr _HeadBase() : _Head() {}

  constexpr _HeadBase(const _Head &__h) : _Head(__h) {}

  constexpr _HeadBase(const _HeadBase &) = default;
  constexpr _HeadBase(_HeadBase &&)      = default;

  template <typename _UHead>
  constexpr _HeadBase(_UHead &&__h) : _Head(std::forward<_UHead>(__h))
  {}

  /*
      _HeadBase(allocator_arg_t, __uses_alloc0)
      : _Head()
      {
      }

      template <typename _Alloc>
      _HeadBase(allocator_arg_t, __uses_alloc1<_Alloc> __a)
      : _Head(allocator_arg, *__a._M_a)
      {
      }

      template <typename _Alloc>
      _HeadBase(allocator_arg_t, __uses_alloc2<_Alloc> __a)
      : _Head(*__a._M_a)
      {
      }

      template <typename _UHead>
      _HeadBase(__uses_alloc0, _UHead&& __uhead)
      : _Head(std::forward<_UHead>(__uhead))
      {
      }

      template <typename _Alloc, typename _UHead>
      _HeadBase(__uses_alloc1<_Alloc> __a, _UHead&& __uhead)
      : _Head(allocator_arg, *__a._M_a, std::forward<_UHead>(__uhead))
      {
      }

      template <typename _Alloc, typename _UHead>
      _HeadBase(__uses_alloc2<_Alloc> __a, _UHead&& __uhead)
      : _Head(std::forward<_UHead>(__uhead), *__a._M_a)
      {
      }
  */
  static constexpr _Head &_M_head(_HeadBase &__b) noexcept { return __b; }

  static constexpr const _Head &_M_head(const _HeadBase &__b) noexcept { return __b; }
};

template <std::size_t _Idx, typename _Head>
struct _HeadBase<_Idx, _Head, false> {
  constexpr _HeadBase() : _M_head_impl() {}

  constexpr _HeadBase(const _Head &__h) : _M_head_impl(__h) {}

  constexpr _HeadBase(const _HeadBase &) = default;
  constexpr _HeadBase(_HeadBase &&)      = default;

  template <typename _UHead>
  constexpr _HeadBase(_UHead &&__h) : _M_head_impl(std::forward<_UHead>(__h))
  {}

  /*
      _HeadBase(allocator_arg_t, __uses_alloc0)
      : _M_head_impl()
      {
      }

      template <typename _Alloc>
      _HeadBase(allocator_arg_t, __uses_alloc1<_Alloc> __a)
      : _M_head_impl(allocator_arg, *__a._M_a)
      {
      }

      template <typename _Alloc>
      _HeadBase(allocator_arg_t, __uses_alloc2<_Alloc> __a)
      : _M_head_impl(*__a._M_a)
      {
      }

      template <typename _UHead>
      _HeadBase(__uses_alloc0, _UHead&& __uhead)
      : _M_head_impl(std::forward<_UHead>(__uhead))
      {
      }

      template <typename _Alloc, typename _UHead>
      _HeadBase(__uses_alloc1<_Alloc> __a, _UHead&& __uhead)
      : _M_head_impl(allocator_arg, *__a._M_a, std::forward<_UHead>(__uhead))
      {
      }

      template <typename _Alloc, typename _UHead>
      _HeadBase(__uses_alloc2<_Alloc> __a, _UHead&& __uhead)
      : _M_head_impl(std::forward<_UHead>(__uhead), *__a._M_a)
      {
      }
  */
  static constexpr _Head &_M_head(_HeadBase &__b) noexcept { return __b._M_head_impl; }
  static constexpr const _Head &_M_head(const _HeadBase &__b) noexcept
  {
    return __b._M_head_impl;
  }

  _Head _M_head_impl;
};

template <typename...>
struct _And;

template <>
struct _And<> : public std::true_type {};

template <typename _B1>
struct _And<_B1> : public _B1 {};

template <typename _B1, typename _B2>
struct _And<_B1, _B2> : public std::conditional<_B1::value, _B2, _B1>::type {};

template <typename _B1, typename _B2, typename _B3, typename... _Bn>
struct _And<_B1, _B2, _B3, _Bn...>
    : public std::conditional<_B1::value, _And<_B2, _B3, _Bn...>, _B1>::type {};

template <bool __v>
using _BoolConstant = std::integral_constant<bool, __v>;

#if __cplusplus > 201402L
#  define __cpp_lib_bool_constant 201505
template <bool __v>
using bool_constant = integral_constant<bool, __v>;
#endif

template <typename _Pp>
struct _Not : public _BoolConstant<!bool(_Pp::value)> {};

/**
 * Contains the actual implementation of the @c tuple template, stored
 * as a recursive inheritance hierarchy from the first element (most
 * derived class) to the last (least derived class). The @c Idx
 * parameter gives the 0-based index of the element stored at this
 * point in the hierarchy; we use it to implement a constant-time
 * Get() operation.
 */
template <std::size_t _Idx, typename... _Elements>
struct _TupleImpl;

/**
 * Recursive tuple implementation. Here we store the @c Head element
 * and derive from a @c Tuple_impl containing the remaining elements
 * (which contains the @c Tail).
 */
template <std::size_t _Idx, typename _Head, typename... _Tail>
struct _TupleImpl<_Idx, _Head, _Tail...> : public _TupleImpl<_Idx + 1, _Tail...>,
                                           private _HeadBase<_Idx, _Head> {
  template <std::size_t, typename...>
  friend struct _TupleImpl;

  typedef _TupleImpl<_Idx + 1, _Tail...> _Inherited;
  typedef _HeadBase<_Idx, _Head> _Base;

  static constexpr _Head &_M_head(_TupleImpl &__t) noexcept
  {
    return _Base::_M_head(__t);
  }
  static constexpr const _Head &_M_head(const _TupleImpl &__t) noexcept
  {
    return _Base::_M_head(__t);
  }
  static constexpr _Inherited &_M_tail(_TupleImpl &__t) noexcept { return __t; }
  static constexpr const _Inherited &_M_tail(const _TupleImpl &__t) noexcept
  {
    return __t;
  }

  constexpr _TupleImpl() : _Inherited(), _Base() {}

  explicit constexpr _TupleImpl(const _Head &__head, const _Tail &... __tail)
      : _Inherited(__tail...), _Base(__head)
  {}

  template <
      typename _UHead, typename... _UTail,
      typename = typename std::enable_if<sizeof...(_Tail) == sizeof...(_UTail)>::type>
  explicit constexpr _TupleImpl(_UHead &&__head, _UTail &&... __tail)
      : _Inherited(std::forward<_UTail>(__tail)...), _Base(std::forward<_UHead>(__head))
  {}

  constexpr _TupleImpl(const _TupleImpl &) = default;

  constexpr _TupleImpl(_TupleImpl &&__in) noexcept(
      _And<std::is_nothrow_move_constructible<_Head>,
           std::is_nothrow_move_constructible<_Inherited>>::value)
      : _Inherited(std::move(_M_tail(__in))), _Base(std::forward<_Head>(_M_head(__in)))
  {}

  template <typename... _UElements>
  constexpr _TupleImpl(const _TupleImpl<_Idx, _UElements...> &__in)
      : _Inherited(_TupleImpl<_Idx, _UElements...>::_M_tail(__in)),
        _Base(_TupleImpl<_Idx, _UElements...>::_M_head(__in))
  {}

  template <typename _UHead, typename... _UTails>
  constexpr _TupleImpl(_TupleImpl<_Idx, _UHead, _UTails...> &&__in)
      : _Inherited(std::move(_TupleImpl<_Idx, _UHead, _UTails...>::_M_tail(__in))),
        _Base(std::forward<_UHead>(_TupleImpl<_Idx, _UHead, _UTails...>::_M_head(__in)))
  {}

  /*
      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a)
      : _Inherited(__tag, __a)
      , _Base(__tag, __use_alloc<_Head>(__a))
      {
      }

      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, const _Head& __head, const
     _Tail&... __tail) : _Inherited(__tag, __a, __tail...) , _Base(__use_alloc<_Head,
     _Alloc, _Head>(__a), __head)
      {
      }

      template <typename _Alloc, typename _UHead, typename... _UTail,
                typename = typename std::enable_if<sizeof...(_Tail) ==
     sizeof...(_UTail)>::type> _TupleImpl(allocator_arg_t __tag, const _Alloc& __a,
     _UHead&& __head, _UTail&&... __tail) : _Inherited(__tag, __a,
     std::forward<_UTail>(__tail)...) , _Base(__use_alloc<_Head, _Alloc, _UHead>(__a),
     std::forward<_UHead>(__head))
      {
      }

      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, const _TupleImpl& __in)
      : _Inherited(__tag, __a, _M_tail(__in))
      , _Base(__use_alloc<_Head, _Alloc, _Head>(__a), _M_head(__in))
      {
      }

      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, _TupleImpl&& __in)
      : _Inherited(__tag, __a, std::move(_M_tail(__in)))
      , _Base(__use_alloc<_Head, _Alloc, _Head>(__a),
     std::forward<_Head>(_M_head(__in)))
      {
      }

      template <typename _Alloc, typename... _UElements>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, const _TupleImpl<_Idx,
     _UElements...>& __in) : _Inherited(__tag, __a, _TupleImpl<_Idx,
     _UElements...>::_M_tail(__in)) , _Base(__use_alloc<_Head, _Alloc, _Head>(__a),
     _TupleImpl<_Idx, _UElements...>::_M_head(__in))
      {
      }

      template <typename _Alloc, typename _UHead, typename... _UTails>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, _TupleImpl<_Idx, _UHead,
     _UTails...>&& __in) : _Inherited(__tag, __a, std::move(_TupleImpl<_Idx, _UHead,
     _UTails...>::_M_tail(__in))) , _Base(__use_alloc<_Head, _Alloc, _UHead>(__a),
              std::forward<_UHead>(_TupleImpl<_Idx, _UHead, _UTails...>::_M_head(__in)))
      {
      }
  */
  _TupleImpl &operator=(const _TupleImpl &__in)
  {
    _M_head(*this) = _M_head(__in);
    _M_tail(*this) = _M_tail(__in);
    return *this;
  }

  _TupleImpl &operator=(_TupleImpl &&__in) noexcept(
      _And<std::is_nothrow_move_assignable<_Head>,
           std::is_nothrow_move_assignable<_Inherited>>::value)
  {
    _M_head(*this) = std::forward<_Head>(_M_head(__in));
    _M_tail(*this) = std::move(_M_tail(__in));
    return *this;
  }

  template <typename... _UElements>
  _TupleImpl &operator=(const _TupleImpl<_Idx, _UElements...> &__in)
  {
    _M_head(*this) = _TupleImpl<_Idx, _UElements...>::_M_head(__in);
    _M_tail(*this) = _TupleImpl<_Idx, _UElements...>::_M_tail(__in);
    return *this;
  }

  template <typename _UHead, typename... _UTails>
  _TupleImpl &operator=(_TupleImpl<_Idx, _UHead, _UTails...> &&__in)
  {
    _M_head(*this) =
        std::forward<_UHead>(_TupleImpl<_Idx, _UHead, _UTails...>::_M_head(__in));
    _M_tail(*this) = std::move(_TupleImpl<_Idx, _UHead, _UTails...>::_M_tail(__in));
    return *this;
  }
  /*
  protected:
      void _M_swap(_TupleImpl& __in) noexcept(
          __is_nothrow_swappable<_Head>::value&&
  noexcept(_M_tail(__in)._M_swap(_M_tail(__in))))
      {
          using std::swap;
          swap(_M_head(*this), _M_head(__in));
          _Inherited::_M_swap(_M_tail(__in));
      }
      */
};

// Basis case of inheritance recursion.
template <std::size_t _Idx, typename _Head>
struct _TupleImpl<_Idx, _Head> : private _HeadBase<_Idx, _Head> {
  template <std::size_t, typename...>
  friend struct _TupleImpl;

  typedef _HeadBase<_Idx, _Head> _Base;

  static constexpr _Head &_M_head(_TupleImpl &__t) noexcept
  {
    return _Base::_M_head(__t);
  }

  static constexpr const _Head &_M_head(const _TupleImpl &__t) noexcept
  {
    return _Base::_M_head(__t);
  }

  constexpr _TupleImpl() : _Base() {}

  explicit constexpr _TupleImpl(const _Head &__head) : _Base(__head) {}

  template <typename _UHead>
  explicit constexpr _TupleImpl(_UHead &&__head) : _Base(std::forward<_UHead>(__head))
  {}

  constexpr _TupleImpl(const _TupleImpl &) = default;

  constexpr _TupleImpl(_TupleImpl &&__in) noexcept(
      std::is_nothrow_move_constructible<_Head>::value)
      : _Base(std::forward<_Head>(_M_head(__in)))
  {}

  template <typename _UHead>
  constexpr _TupleImpl(const _TupleImpl<_Idx, _UHead> &__in)
      : _Base(_TupleImpl<_Idx, _UHead>::_M_head(__in))
  {}

  template <typename _UHead>
  constexpr _TupleImpl(_TupleImpl<_Idx, _UHead> &&__in)
      : _Base(std::forward<_UHead>(_TupleImpl<_Idx, _UHead>::_M_head(__in)))
  {}

  /*
      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a)
      : _Base(__tag, __use_alloc<_Head>(__a))
      {
      }

      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, const _Head& __head)
      : _Base(__use_alloc<_Head, _Alloc, _Head>(__a), __head)
      {
      }

      template <typename _Alloc, typename _UHead>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, _UHead&& __head)
      : _Base(__use_alloc<_Head, _Alloc, _UHead>(__a), std::forward<_UHead>(__head))
      {
      }

      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, const _TupleImpl& __in)
      : _Base(__use_alloc<_Head, _Alloc, _Head>(__a), _M_head(__in))
      {
      }

      template <typename _Alloc>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, _TupleImpl&& __in)
      : _Base(__use_alloc<_Head, _Alloc, _Head>(__a),
     std::forward<_Head>(_M_head(__in)))
      {
      }

      template <typename _Alloc, typename _UHead>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, const _TupleImpl<_Idx,
     _UHead>&
     __in) : _Base(__use_alloc<_Head, _Alloc, _Head>(__a), _TupleImpl<_Idx,
     _UHead>::_M_head(__in))
      {
      }

      template <typename _Alloc, typename _UHead>
      _TupleImpl(allocator_arg_t __tag, const _Alloc& __a, _TupleImpl<_Idx, _UHead>&&
     __in) : _Base(__use_alloc<_Head, _Alloc, _UHead>(__a),
     std::forward<_UHead>(_TupleImpl<_Idx, _UHead>::_M_head(__in)))
      {
      }
  */
  _TupleImpl &operator=(const _TupleImpl &__in)
  {
    _M_head(*this) = _M_head(__in);
    return *this;
  }

  _TupleImpl &operator=(_TupleImpl &&__in) noexcept(
      std::is_nothrow_move_assignable<_Head>::value)
  {
    _M_head(*this) = std::forward<_Head>(_M_head(__in));
    return *this;
  }

  template <typename _UHead>
  _TupleImpl &operator=(const _TupleImpl<_Idx, _UHead> &__in)
  {
    _M_head(*this) = _TupleImpl<_Idx, _UHead>::_M_head(__in);
    return *this;
  }

  template <typename _UHead>
  _TupleImpl &operator=(_TupleImpl<_Idx, _UHead> &&__in)
  {
    _M_head(*this) = std::forward<_UHead>(_TupleImpl<_Idx, _UHead>::_M_head(__in));
    return *this;
  }

protected:
  /*
      void _M_swap(_TupleImpl& __in) noexcept(__is_nothrow_swappable<_Head>::value)
      {
          using std::swap;
          swap(_M_head(*this), _M_head(__in));
      }
      */
};

// Concept utility functions, reused in conditionally-explicit
// constructors.
template <bool, typename... _Elements>
struct _TC {
  template <typename... _UElements>
  static constexpr bool _ConstructibleTuple()
  {
    return _And<std::is_constructible<_Elements, const _UElements &>...>::value;
  }

  template <typename... _UElements>
  static constexpr bool _ImplicitlyConvertibleTuple()
  {
    return _And<std::is_convertible<const _UElements &, _Elements>...>::value;
  }

  template <typename... _UElements>
  static constexpr bool _MoveConstructibleTuple()
  {
    return _And<std::is_constructible<_Elements, _UElements &&>...>::value;
  }

  template <typename... _UElements>
  static constexpr bool _ImplicitlyMoveConvertibleTuple()
  {
    return _And<std::is_convertible<_UElements &&, _Elements>...>::value;
  }

  template <typename _SrcTuple>
  static constexpr bool _NonNestedTuple()
  {
    return _And<_Not<std::is_same<Tuple<_Elements...>,
                                  typename std::remove_cv<typename std::remove_reference<
                                      _SrcTuple>::type>::type>>,
                _Not<std::is_convertible<_SrcTuple, _Elements...>>,
                _Not<std::is_constructible<_Elements..., _SrcTuple>>>::value;
  }
  template <typename... _UElements>
  static constexpr bool __NotSameTuple()
  {
    return _Not<std::is_same<Tuple<_Elements...>,
                             typename std::remove_const<typename std::remove_reference<
                                 _UElements...>::type>::type>>::value;
  }
};

template <typename... _Elements>
struct _TC<false, _Elements...> {
  template <typename... _UElements>
  static constexpr bool _ConstructibleTuple()
  {
    return false;
  }

  template <typename... _UElements>
  static constexpr bool _ImplicitlyConvertibleTuple()
  {
    return false;
  }

  template <typename... _UElements>
  static constexpr bool _MoveConstructibleTuple()
  {
    return false;
  }

  template <typename... _UElements>
  static constexpr bool _ImplicitlyMoveConvertibleTuple()
  {
    return false;
  }

  template <typename... _UElements>
  static constexpr bool _NonNestedTuple()
  {
    return true;
  }
  template <typename... _UElements>
  static constexpr bool __NotSameTuple()
  {
    return true;
  }
};

/// Primary class template, tuple
template <typename... _Elements>
class Tuple : public _TupleImpl<0, _Elements...> {
  typedef _TupleImpl<0, _Elements...> _Inherited;

  // Used for constraining the default constructor so
  // that it becomes dependent on the constraints.
  template <typename _Dummy>
  struct _TC2 {
    static constexpr bool _DefaultConstructibleTuple()
    {
      return _And<std::is_default_constructible<_Elements>...>::value;
    }
    static constexpr bool _ImplicitlyDefaultConstructibleTuple()
    {
      return _And<std::is_trivially_default_constructible<_Elements>...>::value;
    }
  };

public:
  template <typename _Dummy                     = void,
            typename std::enable_if<_TC2<_Dummy>::_ImplicitlyDefaultConstructibleTuple(),
                                    bool>::type = true>
  constexpr Tuple() : _Inherited()
  {}

  template <
      typename _Dummy                     = void,
      typename std::enable_if<_TC2<_Dummy>::_DefaultConstructibleTuple() &&
                                  !_TC2<_Dummy>::_ImplicitlyDefaultConstructibleTuple(),
                              bool>::type = false>
  explicit constexpr Tuple() : _Inherited()
  {}

  // Shortcut for the cases where constructors taking _Elements...
  // need to be constrained.
  template <typename _Dummy>
  using _TCC = _TC<std::is_same<_Dummy, void>::value, _Elements...>;

  template <typename _Dummy = void,
            typename std::enable_if<
                _TCC<_Dummy>::template _ConstructibleTuple<_Elements...>() &&
                    _TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_Elements...>() &&
                    (sizeof...(_Elements) >= 1),
                bool>::type = true>
  constexpr Tuple(const _Elements &... __elements) : _Inherited(__elements...)
  {}

  template <typename _Dummy = void,
            typename std::enable_if<
                _TCC<_Dummy>::template _ConstructibleTuple<_Elements...>() &&
                    !_TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_Elements...>() &&
                    (sizeof...(_Elements) >= 1),
                bool>::type = false>
  explicit constexpr Tuple(const _Elements &... __elements) : _Inherited(__elements...)
  {}

  // Shortcut for the cases where constructors taking _UElements...
  // need to be constrained.
  template <typename... _UElements>
  using _TMC = _TC<(sizeof...(_Elements) == sizeof...(_UElements)) &&
                       (_TC<(sizeof...(_UElements) == 1),
                            _Elements...>::template __NotSameTuple<_UElements...>()),
                   _Elements...>;

  // Shortcut for the cases where constructors taking Tuple<_UElements...>
  // need to be constrained.
  template <typename... _UElements>
  using _TMCT = _TC<(sizeof...(_Elements) == sizeof...(_UElements)) &&
                        !std::is_same<Tuple<_Elements...>, Tuple<_UElements...>>::value,
                    _Elements...>;

  template <typename... _UElements,
            typename std::enable_if<
                _TMC<_UElements...>::template _MoveConstructibleTuple<_UElements...>() &&
                    _TMC<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                        _UElements...>() &&
                    (sizeof...(_Elements) >= 1),
                bool>::type = true>
  constexpr Tuple(_UElements &&... __elements)
      : _Inherited(std::forward<_UElements>(__elements)...)
  {}

  template <typename... _UElements,
            typename std::enable_if<
                _TMC<_UElements...>::template _MoveConstructibleTuple<_UElements...>() &&
                    !_TMC<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                        _UElements...>() &&
                    (sizeof...(_Elements) >= 1),
                bool>::type = false>
  explicit constexpr Tuple(_UElements &&... __elements)
      : _Inherited(std::forward<_UElements>(__elements)...)
  {}

  constexpr Tuple(const Tuple &) = default;

  constexpr Tuple(Tuple &&) = default;

  // Shortcut for the cases where constructors taking Tuples
  // must avoid creating temporaries.
  template <typename _Dummy>
  using _TNTC =
      _TC<std::is_same<_Dummy, void>::value && sizeof...(_Elements) == 1, _Elements...>;

  template <
      typename... _UElements, typename _Dummy = void,
      typename std::enable_if<
          _TMCT<_UElements...>::template _ConstructibleTuple<_UElements...>() &&
              _TMCT<_UElements...>::template _ImplicitlyConvertibleTuple<
                  _UElements...>() &&
              _TNTC<_Dummy>::template _NonNestedTuple<const Tuple<_UElements...> &>(),
          bool>::type = true>
  constexpr Tuple(const Tuple<_UElements...> &__in)
      : _Inherited(static_cast<const _TupleImpl<0, _UElements...> &>(__in))
  {}

  template <
      typename... _UElements, typename _Dummy = void,
      typename std::enable_if<
          _TMCT<_UElements...>::template _ConstructibleTuple<_UElements...>() &&
              !_TMCT<_UElements...>::template _ImplicitlyConvertibleTuple<
                  _UElements...>() &&
              _TNTC<_Dummy>::template _NonNestedTuple<const Tuple<_UElements...> &>(),
          bool>::type = false>
  explicit constexpr Tuple(const Tuple<_UElements...> &__in)
      : _Inherited(static_cast<const _TupleImpl<0, _UElements...> &>(__in))
  {}

  template <typename... _UElements, typename _Dummy = void,
            typename std::enable_if<
                _TMCT<_UElements...>::template _MoveConstructibleTuple<_UElements...>() &&
                    _TMCT<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                        _UElements...>() &&
                    _TNTC<_Dummy>::template _NonNestedTuple<Tuple<_UElements...> &&>(),
                bool>::type = true>
  constexpr Tuple(Tuple<_UElements...> &&__in)
      : _Inherited(static_cast<_TupleImpl<0, _UElements...> &&>(__in))
  {}

  template <typename... _UElements, typename _Dummy = void,
            typename std::enable_if<
                _TMCT<_UElements...>::template _MoveConstructibleTuple<_UElements...>() &&
                    !_TMCT<_UElements...>::template _ImplicitlyMoveConvertibleTuple<
                        _UElements...>() &&
                    _TNTC<_Dummy>::template _NonNestedTuple<Tuple<_UElements...> &&>(),
                bool>::type = false>
  explicit constexpr Tuple(Tuple<_UElements...> &&__in)
      : _Inherited(static_cast<_TupleImpl<0, _UElements...> &&>(__in))
  {}

  // Allocator-extended constructors.
  /*
      template <typename _Alloc>
      Tuple(allocator_arg_t __tag, const _Alloc& __a)
      : _Inherited(__tag, __a)
      {
      }

      template <typename _Alloc, typename _Dummy = void,
                typename std::enable_if<_TCC<_Dummy>::template
     _ConstructibleTuple<_Elements...>() && _TCC<_Dummy>::template
     _ImplicitlyConvertibleTuple<_Elements...>(), bool>::type = true>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, const _Elements&... __elements)
      : _Inherited(__tag, __a, __elements...)
      {
      }

      template <typename _Alloc, typename _Dummy = void,
                typename std::enable_if<_TCC<_Dummy>::template
     _ConstructibleTuple<_Elements...>() &&
                                       !_TCC<_Dummy>::template
     _ImplicitlyConvertibleTuple<_Elements...>(), bool>::type = false> explicit
     Tuple(allocator_arg_t __tag, const _Alloc& __a, const _Elements&... __elements) :
     _Inherited(__tag, __a, __elements...)
      {
      }

      template <typename _Alloc, typename... _UElements,
                typename std::enable_if<_TMC<_UElements...>::template
     _MoveConstructibleTuple<_UElements...>() && _TMC<_UElements...>::template
     _ImplicitlyMoveConvertibleTuple<_UElements...>(), bool>::type = true>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, _UElements&&... __elements)
      : _Inherited(__tag, __a, std::forward<_UElements>(__elements)...)
      {
      }

      template <typename _Alloc, typename... _UElements,
                typename std::enable_if<_TMC<_UElements...>::template
     _MoveConstructibleTuple<_UElements...>() &&
                                       !_TMC<_UElements...>::template
     _ImplicitlyMoveConvertibleTuple<_UElements...>(), bool>::type = false> explicit
     Tuple(allocator_arg_t __tag, const _Alloc& __a, _UElements&&... __elements) :
     _Inherited(__tag, __a, std::forward<_UElements>(__elements)...)
      {
      }

      template <typename _Alloc>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, const Tuple& __in)
      : _Inherited(__tag, __a, static_cast<const _Inherited&>(__in))
      {
      }

      template <typename _Alloc>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, Tuple&& __in)
      : _Inherited(__tag, __a, static_cast<_Inherited&&>(__in))
      {
      }

      template <typename _Alloc, typename _Dummy = void, typename... _UElements,
                typename std::enable_if<_TMCT<_UElements...>::template
     _ConstructibleTuple<_UElements...>() && _TMCT<_UElements...>::template
     _ImplicitlyConvertibleTuple<_UElements...>() && _TNTC<_Dummy>::template
     _NonNestedTuple<Tuple<_UElements...>&&>(), bool>::type = true>
     Tuple(allocator_arg_t
     __tag, const _Alloc& __a, const Tuple<_UElements...>& __in) : _Inherited(__tag,
     __a, static_cast<const _TupleImpl<0, _UElements...>&>(__in))
      {
      }

      template <typename _Alloc, typename _Dummy = void, typename... _UElements,
                typename std::enable_if<_TMCT<_UElements...>::template
     _ConstructibleTuple<_UElements...>() &&
                                       !_TMCT<_UElements...>::template
     _ImplicitlyConvertibleTuple<_UElements...>() && _TNTC<_Dummy>::template
     _NonNestedTuple<Tuple<_UElements...>&&>(), bool>::type = false> explicit
     Tuple(allocator_arg_t __tag, const _Alloc& __a, const Tuple<_UElements...>& __in) :
     _Inherited(__tag, __a, static_cast<const _TupleImpl<0, _UElements...>&>(__in))
      {
      }

      template <typename _Alloc, typename _Dummy = void, typename... _UElements,
                typename std::enable_if<_TMCT<_UElements...>::template
     _MoveConstructibleTuple<_UElements...>() && _TMCT<_UElements...>::template
     _ImplicitlyMoveConvertibleTuple<_UElements...>()
     && _TNTC<_Dummy>::template _NonNestedTuple<Tuple<_UElements...>&&>(), bool>::type =
     true> Tuple(allocator_arg_t
     __tag, const _Alloc& __a, Tuple<_UElements...>&& __in) : _Inherited(__tag, __a,
     static_cast<_TupleImpl<0, _UElements...>&&>(__in))
      {
      }

      template <typename _Alloc, typename _Dummy = void, typename... _UElements,
                typename std::enable_if<_TMCT<_UElements...>::template
     _MoveConstructibleTuple<_UElements...>() &&
                                       !_TMCT<_UElements...>::template
     _ImplicitlyMoveConvertibleTuple<_UElements...>() && _TNTC<_Dummy>::template
     _NonNestedTuple<Tuple<_UElements...>&&>(), bool>::type = false> explicit
     Tuple(allocator_arg_t __tag, const _Alloc& __a, Tuple<_UElements...>&& __in) :
     _Inherited(__tag, __a, static_cast<_TupleImpl<0, _UElements...>&&>(__in))
      {
      }
  */
  Tuple &operator=(const Tuple &__in)
  {
    static_cast<_Inherited &>(*this) = __in;
    return *this;
  }

  Tuple &operator=(Tuple &&__in) noexcept(
      std::is_nothrow_move_assignable<_Inherited>::value)
  {
    static_cast<_Inherited &>(*this) = std::move(__in);
    return *this;
  }

  template <typename... _UElements>
  typename std::enable_if<sizeof...(_UElements) == sizeof...(_Elements), Tuple &>::type
  operator=(const Tuple<_UElements...> &__in)
  {
    static_cast<_Inherited &>(*this) = __in;
    return *this;
  }

  template <typename... _UElements>
  typename std::enable_if<sizeof...(_UElements) == sizeof...(_Elements), Tuple &>::type
  operator=(Tuple<_UElements...> &&__in)
  {
    static_cast<_Inherited &>(*this) = std::move(__in);
    return *this;
  }

  void swap(Tuple &__in) noexcept(noexcept(__in._M_swap(__in)))
  {
    _Inherited::_M_swap(__in);
  }
};

#if __cpp_deduction_guides >= 201606
template <typename... _UTypes>
Tuple(_UTypes...)->Tuple<_UTypes...>;
template <typename _T1, typename _T2>
Tuple(pair<_T1, _T2>)->Tuple<_T1, _T2>;
template <typename _Alloc, typename... _UTypes>
Tuple(allocator_arg_t, _Alloc, _UTypes...)->Tuple<_UTypes...>;
template <typename _Alloc, typename _T1, typename _T2>
Tuple(allocator_arg_t, _Alloc, pair<_T1, _T2>)->Tuple<_T1, _T2>;
template <typename _Alloc, typename... _UTypes>
Tuple(allocator_arg_t, _Alloc, Tuple<_UTypes...>)->Tuple<_UTypes...>;
#endif

// Explicit specialization, zero-element Tuple.
template <>
class Tuple<> {
public:
  void swap(Tuple &) noexcept
  { /* no-op */
  }
  // We need the default since we're going to define no-op
  // allocator constructors.
  Tuple() = default;
  // No-op allocator constructors.
  /*
  template <typename _Alloc>
  Tuple(allocator_arg_t, const _Alloc&)
  {
  }
  template <typename _Alloc>
  Tuple(allocator_arg_t, const _Alloc&, const Tuple&)
  {
  }
  */
};

/// Partial specialization, 2-element Tuple.
/// Includes construction and assignment from a pair.
template <typename _T1, typename _T2>
class Tuple<_T1, _T2> : public _TupleImpl<0, _T1, _T2> {
  typedef _TupleImpl<0, _T1, _T2> _Inherited;

public:
  template <
      typename _U1 = _T1, typename _U2 = _T2,
      typename std::enable_if<_And<std::is_trivially_default_constructible<_U1>,
                                   std::is_trivially_default_constructible<_U2>>::value,
                              bool>::type = true>

  constexpr Tuple() : _Inherited()
  {}

  template <
      typename _U1 = _T1, typename _U2 = _T2,
      typename std::enable_if<
          _And<std::is_default_constructible<_U1>, std::is_default_constructible<_U2>,
               _Not<_And<std::is_trivially_default_constructible<_U1>,
                         std::is_trivially_default_constructible<_U2>>>>::value,
          bool>::type = false>

  explicit constexpr Tuple() : _Inherited()
  {}

  // Shortcut for the cases where constructors taking _T1, _T2
  // need to be constrained.
  template <typename _Dummy>
  using _TCC = _TC<std::is_same<_Dummy, void>::value, _T1, _T2>;

  template <typename _Dummy = void,
            typename std::enable_if<
                _TCC<_Dummy>::template _ConstructibleTuple<_T1, _T2>() &&
                    _TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_T1, _T2>(),
                bool>::type = true>
  constexpr Tuple(const _T1 &__a1, const _T2 &__a2) : _Inherited(__a1, __a2)
  {}

  template <typename _Dummy = void,
            typename std::enable_if<
                _TCC<_Dummy>::template _ConstructibleTuple<_T1, _T2>() &&
                    !_TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_T1, _T2>(),
                bool>::type = false>
  explicit constexpr Tuple(const _T1 &__a1, const _T2 &__a2) : _Inherited(__a1, __a2)
  {}

  // Shortcut for the cases where constructors taking _U1, _U2
  // need to be constrained.
  using _TMC = _TC<true, _T1, _T2>;

  /*
      template <typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>() &&
                                            !std::is_same<typename
     std::decay<_U1>::type, allocator_arg_t>::value, bool>::type = true> constexpr
     Tuple(_U1&& __a1, _U2&& __a2) : _Inherited(std::forward<_U1>(__a1),
     std::forward<_U2>(__a2))
      {
      }

      template <typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     &&
                                            !_TMC::template
     _ImplicitlyMoveConvertibleTuple<_U1, _U2>() && !std::is_same<typename
     decay<_U1>::type, allocator_arg_t>::value, bool>::type = false> explicit constexpr
     Tuple(_U1&& __a1, _U2&& __a2) : _Inherited(std::forward<_U1>(__a1),
     std::forward<_U2>(__a2))
      {
      }
  */
  constexpr Tuple(const Tuple &) = default;

  constexpr Tuple(Tuple &&) = default;

  template <
      typename _U1, typename _U2,
      typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>() &&
                                  _TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                              bool>::type = true>
  constexpr Tuple(const Tuple<_U1, _U2> &__in)
      : _Inherited(static_cast<const _TupleImpl<0, _U1, _U2> &>(__in))
  {}

  template <
      typename _U1, typename _U2,
      typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>() &&
                                  !_TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                              bool>::type = false>
  explicit constexpr Tuple(const Tuple<_U1, _U2> &__in)
      : _Inherited(static_cast<const _TupleImpl<0, _U1, _U2> &>(__in))
  {}

  template <typename _U1, typename _U2,
            typename std::enable_if<
                _TMC::template _MoveConstructibleTuple<_U1, _U2>() &&
                    _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                bool>::type = true>
  constexpr Tuple(Tuple<_U1, _U2> &&__in)
      : _Inherited(static_cast<_TupleImpl<0, _U1, _U2> &&>(__in))
  {}

  template <typename _U1, typename _U2,
            typename std::enable_if<
                _TMC::template _MoveConstructibleTuple<_U1, _U2>() &&
                    !_TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                bool>::type = false>
  explicit constexpr Tuple(Tuple<_U1, _U2> &&__in)
      : _Inherited(static_cast<_TupleImpl<0, _U1, _U2> &&>(__in))
  {}

  template <
      typename _U1, typename _U2,
      typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>() &&
                                  _TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                              bool>::type = true>
  constexpr Tuple(const std::pair<_U1, _U2> &__in) : _Inherited(__in.first, __in.second)
  {}

  template <
      typename _U1, typename _U2,
      typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>() &&
                                  !_TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(),
                              bool>::type = false>
  explicit constexpr Tuple(const std::pair<_U1, _U2> &__in)
      : _Inherited(__in.first, __in.second)
  {}

  template <typename _U1, typename _U2,
            typename std::enable_if<
                _TMC::template _MoveConstructibleTuple<_U1, _U2>() &&
                    _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                bool>::type = true>
  constexpr Tuple(std::pair<_U1, _U2> &&__in)
      : _Inherited(std::forward<_U1>(__in.first), std::forward<_U2>(__in.second))
  {}

  template <typename _U1, typename _U2,
            typename std::enable_if<
                _TMC::template _MoveConstructibleTuple<_U1, _U2>() &&
                    !_TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(),
                bool>::type = false>
  explicit constexpr Tuple(std::pair<_U1, _U2> &&__in)
      : _Inherited(std::forward<_U1>(__in.first), std::forward<_U2>(__in.second))
  {}

  // Allocator-extended constructors.
  /*
      template <typename _Alloc>
      Tuple(allocator_arg_t __tag, const _Alloc& __a)
      : _Inherited(__tag, __a)
      {
      }

      template <typename _Alloc, typename _Dummy = void,
                typename std::enable_if<_TCC<_Dummy>::template _ConstructibleTuple<_T1,
     _T2>() && _TCC<_Dummy>::template _ImplicitlyConvertibleTuple<_T1, _T2>(),
     bool>::type = true>

      Tuple(allocator_arg_t __tag, const _Alloc& __a, const _T1& __a1, const _T2& __a2)
      : _Inherited(__tag, __a, __a1, __a2)
      {
      }

      template <typename _Alloc, typename _Dummy = void,
                typename std::enable_if<_TCC<_Dummy>::template _ConstructibleTuple<_T1,
     _T2>() &&
                                       !_TCC<_Dummy>::template
     _ImplicitlyConvertibleTuple<_T1, _T2>(), bool>::type = false>

      explicit Tuple(allocator_arg_t __tag, const _Alloc& __a, const _T1& __a1, const
     _T2&
     __a2) : _Inherited(__tag, __a, __a1, __a2)
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(), bool>::type = true>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, _U1&& __a1, _U2&& __a2)
      : _Inherited(__tag, __a, std::forward<_U1>(__a1), std::forward<_U2>(__a2))
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     &&
                                       !_TMC::template
     _ImplicitlyMoveConvertibleTuple<_U1, _U2>(), bool>::type = false> explicit
     Tuple(allocator_arg_t __tag, const _Alloc& __a, _U1&& __a1, _U2&& __a2) :
     _Inherited(__tag, __a, std::forward<_U1>(__a1), std::forward<_U2>(__a2))
      {
      }

      template <typename _Alloc>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, const Tuple& __in)
      : _Inherited(__tag, __a, static_cast<const _Inherited&>(__in))
      {
      }

      template <typename _Alloc>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, Tuple&& __in)
      : _Inherited(__tag, __a, static_cast<_Inherited&&>(__in))
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
     && _TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(), bool>::type = true>
     Tuple(allocator_arg_t __tag, const _Alloc& __a, const Tuple<_U1, _U2>& __in) :
     _Inherited(__tag, __a, static_cast<const _TupleImpl<0, _U1, _U2>&>(__in))
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
     &&
                                       !_TMC::template _ImplicitlyConvertibleTuple<_U1,
     _U2>(), bool>::type = false> explicit Tuple(allocator_arg_t __tag, const _Alloc&
     __a, const Tuple<_U1, _U2>& __in) : _Inherited(__tag, __a, static_cast<const
     _TupleImpl<0, _U1, _U2>&>(__in))
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(), bool>::type = true>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, Tuple<_U1, _U2>&& __in)
      : _Inherited(__tag, __a, static_cast<_TupleImpl<0, _U1, _U2>&&>(__in))
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     &&
                                       !_TMC::template
     _ImplicitlyMoveConvertibleTuple<_U1, _U2>(), bool>::type = false> explicit
     Tuple(allocator_arg_t __tag, const _Alloc& __a, Tuple<_U1, _U2>&& __in) :
     _Inherited(__tag, __a, static_cast<_TupleImpl<0, _U1, _U2>&&>(__in))
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
     && _TMC::template _ImplicitlyConvertibleTuple<_U1, _U2>(), bool>::type = true>
     Tuple(allocator_arg_t __tag, const _Alloc& __a, const pair<_U1, _U2>& __in) :
     _Inherited(__tag, __a, __in.first, __in.second)
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _ConstructibleTuple<_U1, _U2>()
     &&
                                       !_TMC::template _ImplicitlyConvertibleTuple<_U1,
     _U2>(), bool>::type = false> explicit Tuple(allocator_arg_t __tag, const _Alloc&
     __a, const pair<_U1, _U2>& __in) : _Inherited(__tag, __a, __in.first, __in.second)
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     && _TMC::template _ImplicitlyMoveConvertibleTuple<_U1, _U2>(), bool>::type = true>
      Tuple(allocator_arg_t __tag, const _Alloc& __a, pair<_U1, _U2>&& __in)
      : _Inherited(__tag, __a, std::forward<_U1>(__in.first),
     std::forward<_U2>(__in.second))
      {
      }

      template <typename _Alloc, typename _U1, typename _U2,
                typename std::enable_if<_TMC::template _MoveConstructibleTuple<_U1,
     _U2>()
     &&
                                       !_TMC::template
     _ImplicitlyMoveConvertibleTuple<_U1, _U2>(), bool>::type = false> explicit
     Tuple(allocator_arg_t __tag, const _Alloc& __a, pair<_U1, _U2>&& __in) :
     _Inherited(__tag, __a, std::forward<_U1>(__in.first),
     std::forward<_U2>(__in.second))
      {
      }
  */
  Tuple &operator=(const Tuple &__in)
  {
    static_cast<_Inherited &>(*this) = __in;
    return *this;
  }

  Tuple &operator=(Tuple &&__in) noexcept(
      std::is_nothrow_move_assignable<_Inherited>::value)
  {
    static_cast<_Inherited &>(*this) = std::move(__in);
    return *this;
  }

  template <typename _U1, typename _U2>
  Tuple &operator=(const Tuple<_U1, _U2> &__in)
  {
    static_cast<_Inherited &>(*this) = __in;
    return *this;
  }

  template <typename _U1, typename _U2>
  Tuple &operator=(Tuple<_U1, _U2> &&__in)
  {
    static_cast<_Inherited &>(*this) = std::move(__in);
    return *this;
  }

  template <typename _U1, typename _U2>
  Tuple &operator=(const std::pair<_U1, _U2> &__in)
  {
    this->_M_head(*this)                = __in.first;
    this->_M_tail(*this)._M_head(*this) = __in.second;
    return *this;
  }

  template <typename _U1, typename _U2>
  Tuple &operator=(std::pair<_U1, _U2> &&__in)
  {
    this->_M_head(*this)                = std::forward<_U1>(__in.first);
    this->_M_tail(*this)._M_head(*this) = std::forward<_U2>(__in.second);
    return *this;
  }

  void swap(Tuple &__in) noexcept(noexcept(__in._M_swap(__in)))
  {
    _Inherited::_M_swap(__in);
  }
};

/// class TupleSize
template <typename... _Elements>
struct TupleSize<Tuple<_Elements...>>
    : public std::integral_constant<std::size_t, sizeof...(_Elements)> {};

#if __cplusplus > 201402L
template <typename _Tp>
inline constexpr size_t TupleSize_v = TupleSize<_Tp>::value;
#endif

/**
 * Recursive case for TupleElement: strip off the first element in
 * the Tuple and retrieve the (i-1)th element of the remaining Tuple.
 */
template <std::size_t __i, typename _Head, typename... _Tail>
struct TupleElement<__i, Tuple<_Head, _Tail...>>
    : TupleElement<__i - 1, Tuple<_Tail...>> {};

/**
 * Basis case for TupleElement: The first element is the one we're seeking.
 */
template <typename _Head, typename... _Tail>
struct TupleElement<0, Tuple<_Head, _Tail...>> {
  typedef _Head type;
};

/**
 * Error case for TupleElement: invalid index.
 */
template <size_t __i>
struct TupleElement<__i, Tuple<>> {
  static_assert(__i < TupleSize<Tuple<>>::value, "Tuple index is in range");
};

template <std::size_t __i, typename _Head, typename... _Tail>
constexpr _Head &_GetHelper(_TupleImpl<__i, _Head, _Tail...> &__t) noexcept
{
  return _TupleImpl<__i, _Head, _Tail...>::_M_head(__t);
}

template <std::size_t __i, typename _Head, typename... _Tail>
constexpr const _Head &_GetHelper(const _TupleImpl<__i, _Head, _Tail...> &__t) noexcept
{
  return _TupleImpl<__i, _Head, _Tail...>::_M_head(__t);
}

/// Return a reference to the ith element of a Tuple.
template <std::size_t __i, typename... _Elements>
constexpr __TupleElement_t<__i, Tuple<_Elements...>> &Get(
    Tuple<_Elements...> &__t) noexcept
{
  return _GetHelper<__i>(__t);
}

/// Return a const reference to the ith element of a const Tuple.
template <std::size_t __i, typename... _Elements>
constexpr const __TupleElement_t<__i, Tuple<_Elements...>> &Get(
    const Tuple<_Elements...> &__t) noexcept
{
  return _GetHelper<__i>(__t);
}

/// Return an rvalue reference to the ith element of a Tuple rvalue.
template <std::size_t __i, typename... _Elements>
constexpr __TupleElement_t<__i, Tuple<_Elements...>> &&Get(
    Tuple<_Elements...> &&__t) noexcept
{
  typedef __TupleElement_t<__i, Tuple<_Elements...>> __element_type;
  return std::forward<__element_type &&>(Get<__i>(__t));
}

template <typename _Head, size_t __i, typename... _Tail>
constexpr _Head &_GetHelper2(_TupleImpl<__i, _Head, _Tail...> &__t) noexcept
{
  return _TupleImpl<__i, _Head, _Tail...>::_M_head(__t);
}

template <typename _Head, size_t __i, typename... _Tail>
constexpr const _Head &_GetHelper2(const _TupleImpl<__i, _Head, _Tail...> &__t) noexcept
{
  return _TupleImpl<__i, _Head, _Tail...>::_M_head(__t);
}

/// Return a reference to the unique element of type _Tp of a Tuple.
template <typename _Tp, typename... _Types>
constexpr _Tp &Get(Tuple<_Types...> &__t) noexcept
{
  return _GetHelper2<_Tp>(__t);
}

/// Return a reference to the unique element of type _Tp of a Tuple rvalue.
template <typename _Tp, typename... _Types>
constexpr _Tp &&Get(Tuple<_Types...> &&__t) noexcept
{
  return std::forward<_Tp &&>(_GetHelper2<_Tp>(__t));
}

/// Return a const reference to the unique element of type _Tp of a Tuple.
template <typename _Tp, typename... _Types>
constexpr const _Tp &Get(const Tuple<_Types...> &__t) noexcept
{
  return _GetHelper2<_Tp>(__t);
}

// This class performs the comparison operations on Tuples
template <typename _Tp, typename _Up, size_t __i, size_t __size>
struct _TupleCompare {
  static constexpr bool __eq(const _Tp &__t, const _Up &__u)
  {
    return bool(Get<__i>(__t) == Get<__i>(__u)) &&
           _TupleCompare<_Tp, _Up, __i + 1, __size>::__eq(__t, __u);
  }

  static constexpr bool __less(const _Tp &__t, const _Up &__u)
  {
    return bool(Get<__i>(__t) < Get<__i>(__u)) ||
           (!bool(Get<__i>(__u) < Get<__i>(__t)) &&
            _TupleCompare<_Tp, _Up, __i + 1, __size>::__less(__t, __u));
  }
};

template <typename _Tp, typename _Up, size_t __size>
struct _TupleCompare<_Tp, _Up, __size, __size> {
  static constexpr bool __eq(const _Tp &, const _Up &) { return true; }

  static constexpr bool __less(const _Tp &, const _Up &) { return false; }
};

template <typename... _TElements, typename... _UElements>
constexpr bool operator==(const Tuple<_TElements...> &__t,
                          const Tuple<_UElements...> &__u)
{
  static_assert(sizeof...(_TElements) == sizeof...(_UElements),
                "Tuple objects can only be compared if they have equal sizes.");
  using __compare =
      _TupleCompare<Tuple<_TElements...>, Tuple<_UElements...>, 0, sizeof...(_TElements)>;
  return __compare::__eq(__t, __u);
}

template <typename... _TElements, typename... _UElements>
constexpr bool operator<(const Tuple<_TElements...> &__t, const Tuple<_UElements...> &__u)
{
  static_assert(sizeof...(_TElements) == sizeof...(_UElements),
                "Tuple objects can only be compared if they have equal sizes.");
  using __compare =
      _TupleCompare<Tuple<_TElements...>, Tuple<_UElements...>, 0, sizeof...(_TElements)>;
  return __compare::__less(__t, __u);
}

template <typename... _TElements, typename... _UElements>
constexpr bool operator!=(const Tuple<_TElements...> &__t,
                          const Tuple<_UElements...> &__u)
{
  return !(__t == __u);
}

template <typename... _TElements, typename... _UElements>
constexpr bool operator>(const Tuple<_TElements...> &__t, const Tuple<_UElements...> &__u)
{
  return __u < __t;
}

template <typename... _TElements, typename... _UElements>
constexpr bool operator<=(const Tuple<_TElements...> &__t,
                          const Tuple<_UElements...> &__u)
{
  return !(__u < __t);
}

template <typename... _TElements, typename... _UElements>
constexpr bool operator>=(const Tuple<_TElements...> &__t,
                          const Tuple<_UElements...> &__u)
{
  return !(__t < __u);
}

// NB: DR 705.
template <typename... _Elements>
constexpr Tuple<typename _DecayAndStrip<_Elements>::__type...> MakeTuple(
    _Elements &&... __args)
{
  typedef Tuple<typename _DecayAndStrip<_Elements>::__type...> __result_type;
  return __result_type(std::forward<_Elements>(__args)...);
}

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// 2275. Why is ForwardAsTuple not constexpr?
template <typename... _Elements>
constexpr Tuple<_Elements &&...> ForwardAsTuple(_Elements &&... __args) noexcept
{
  return Tuple<_Elements &&...>(std::forward<_Elements>(__args)...);
}

template <size_t, typename, typename, size_t>
struct _MakeTupleImpl;

template <size_t _Idx, typename _Tuple, typename... _Tp, size_t _Nm>
struct _MakeTupleImpl<_Idx, Tuple<_Tp...>, _Tuple, _Nm>
    : _MakeTupleImpl<_Idx + 1, Tuple<_Tp..., __TupleElement_t<_Idx, _Tuple>>, _Tuple,
                     _Nm> {};

template <std::size_t _Nm, typename _Tuple, typename... _Tp>
struct _MakeTupleImpl<_Nm, Tuple<_Tp...>, _Tuple, _Nm> {
  typedef Tuple<_Tp...> __type;
};

template <typename _Tuple>
struct _DoMakeTuple : _MakeTupleImpl<0, Tuple<>, _Tuple, TupleSize<_Tuple>::value> {};

// Returns the std::Tuple equivalent of a Tuple-like type.
template <typename _Tuple>
struct __MakeTuple
    : public _DoMakeTuple<
          typename std::remove_cv<typename std::remove_reference<_Tuple>::type>::type> {};

// Combines several std::Tuple's into a single one.
template <typename...>
struct _CombineTuples;

template <>
struct _CombineTuples<> {
  typedef Tuple<> __type;
};

template <typename... _Ts>
struct _CombineTuples<Tuple<_Ts...>> {
  typedef Tuple<_Ts...> __type;
};

template <typename... _T1s, typename... _T2s, typename... _Rem>
struct _CombineTuples<Tuple<_T1s...>, Tuple<_T2s...>, _Rem...> {
  typedef typename _CombineTuples<Tuple<_T1s..., _T2s...>, _Rem...>::__type __type;
};

// Computes the result type of TupleCat given a set of Tuple-like types.
template <typename... _Tpls>
struct TupleCatResult {
  typedef typename _CombineTuples<typename __MakeTuple<_Tpls>::__type...>::__type __type;
};

// Helper to determine the index set for the first Tuple-like
// type of a given set.
template <typename...>
struct _Make1stIndices;

template <>
struct _Make1stIndices<> {
  typedef _IndexTuple<> __type;
};

template <typename _Tp, typename... _Tpls>
struct _Make1stIndices<_Tp, _Tpls...> {
  typedef typename _BuildIndexTuple<
      TupleSize<typename std::remove_reference<_Tp>::type>::value>::__type __type;
};

// Performs the actual concatenation by step-wise expanding Tuple-like
// objects into the elements,  which are finally forwarded into the
// result Tuple.
template <typename _Ret, typename _Indices, typename... _Tpls>
struct TupleConcater;

template <typename _Ret, std::size_t... _Is, typename _Tp, typename... _Tpls>
struct TupleConcater<_Ret, _IndexTuple<_Is...>, _Tp, _Tpls...> {
  template <typename... _Us>
  static constexpr _Ret _S_do(_Tp &&__tp, _Tpls &&... __tps, _Us &&... __us)
  {
    typedef typename _Make1stIndices<_Tpls...>::__type __idx;
    typedef TupleConcater<_Ret, __idx, _Tpls...> __next;
    return __next::_S_do(std::forward<_Tpls>(__tps)..., std::forward<_Us>(__us)...,
                         Get<_Is>(std::forward<_Tp>(__tp))...);
  }
};

template <typename _Ret>
struct TupleConcater<_Ret, _IndexTuple<>> {
  template <typename... _Us>
  static constexpr _Ret _S_do(_Us &&... __us)
  {
    return _Ret(std::forward<_Us>(__us)...);
  }
};

template <typename>
struct IsTupleLikeImpl : std::false_type {};

template <typename... _Tps>
struct IsTupleLikeImpl<Tuple<_Tps...>> : std::true_type {};

// Internal type trait that allows us to sfinae-protect TupleCat.
template <typename _Tp>
struct IsTupleLike : public IsTupleLikeImpl<typename std::remove_cv<
                         typename std::remove_reference<_Tp>::type>::type>::type {};

/// TupleCat
template <typename... _Tpls,
          typename = typename std::enable_if<_And<IsTupleLike<_Tpls>...>::value>::type>
constexpr auto TupleCat(_Tpls &&... __tpls) -> typename TupleCatResult<_Tpls...>::__type
{
  typedef typename TupleCatResult<_Tpls...>::__type __ret;
  typedef typename _Make1stIndices<_Tpls...>::__type __idx;
  typedef TupleConcater<__ret, __idx, _Tpls...> __concater;
  return __concater::_S_do(std::forward<_Tpls>(__tpls)...);
}

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// 2301. Why is tie not constexpr?
/// tie
template <typename... _Elements>
constexpr Tuple<_Elements &...> tie(_Elements &... __args) noexcept
{
  return Tuple<_Elements &...>(__args...);
}

/// swap
template <typename... _Elements>
inline
    //#if __cplusplus > 201402L || !defined(__STRICT_ANSI__)  // c++1z or gnu++11
    //    // Constrained free swap overload, see p0185r1
    //    typename std::enable_if<_And<__is_swappable<_Elements>...>::value>::type
    //#else
    void
    //#endif
    swap(Tuple<_Elements...> &__x,
         Tuple<_Elements...> &__y) noexcept(noexcept(__x.swap(__y)))
{
  __x.swap(__y);
}

//#if __cplusplus > 201402L || !defined(__STRICT_ANSI__)  // c++1z or gnu++11
// template <typename... _Elements>
// typename std::enable_if<!_And<__is_swappable<_Elements>...>::value>::type
// swap(Tuple<_Elements...>&,
//                                                                            Tuple<_Elements...>&)
//                                                                            = delete;
//#endif

// A class (and instance) which can be used in 'tie' when an element
// of a Tuple is not required.
// _GLIBCXX14_CONSTEXPR
// 2933. PR for LWG 2773 could be clearer
struct _Swallow_assign {
  template <typename _Tp>
  const _Swallow_assign &operator=(const _Tp &) const
  {
    return *this;
  }
};

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// 2773. Making std::ignore constexpr
constexpr _Swallow_assign ignore{};

template <typename _Tp>
using TupleSize_v = typename TupleSize<_Tp>::value;

/*
/// Partial specialization for Tuples
template <typename... _Types, typename _Alloc>
struct uses_allocator<Tuple<_Types...>, _Alloc> : true_type
{
};

// See stl_pair.h...
template <typename _T1, typename _T2>
template <typename... _Args1, typename... _Args2>
inline pair<_T1, _T2>::pair(piecewise_construct_t, Tuple<_Args1...> __first,
Tuple<_Args2...> __second) : pair(__first, __second, typename
_BuildIndexTuple<sizeof...(_Args1)>::__type(), typename
_BuildIndexTuple<sizeof...(_Args2)>::__type())
{
}

template <typename _T1, typename _T2>
template <typename... _Args1, std::size_t... _Indexes1, typename... _Args2, std::size_t...
_Indexes2> inline pair<_T1, _T2>::pair(Tuple<_Args1...>& __Tuple1, Tuple<_Args2...>&
__Tuple2, _IndexTuple<_Indexes1...>, _IndexTuple<_Indexes2...>) :
first(std::forward<_Args1>(Get<_Indexes1>(__Tuple1))...) ,
second(std::forward<_Args2>(Get<_Indexes2>(__Tuple2))...)
{
}
*/
/*
#if __cplusplus > 201402L
#    define __cpp_lib_apply 201603

template <typename _Fn, typename _Tuple, size_t... _Idx>
constexpr decltype(auto) _ApplyImpl(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
{
    return std::__invoke(std::forward<_Fn>(__f), Get<_Idx>(std::forward<_Tuple>(__t))...);
}

template <typename _Fn, typename _Tuple>
constexpr decltype(auto) apply(_Fn&& __f, _Tuple&& __t)
{
    using _Indices = make_index_sequence<TupleSize_v<decay_t<_Tuple>>>;
    return std::_ApplyImpl(std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
}

#    define __cpp_lib_make_from_Tuple 201606

template <typename _Tp, typename _Tuple, size_t... _Idx>
constexpr _Tp __make_from_TupleImpl(_Tuple&& __t, index_sequence<_Idx...>)
{
    return _Tp(Get<_Idx>(std::forward<_Tuple>(__t))...);
}

template <typename _Tp, typename _Tuple>
constexpr _Tp make_from_Tuple(_Tuple&& __t)
{
    return __make_from_TupleImpl<_Tp>(std::forward<_Tuple>(__t),
make_index_sequence<TupleSize_v<decay_t<_Tuple>>>{});
}
#endif  // C++17
*/

//======================================================================================//

template <typename T, typename... Ts>
auto head(Tuple<T, Ts...> t)
{
  return Get<0>(t);
}

template <std::size_t... Ns, typename... Ts>
auto tail_impl(std::index_sequence<Ns...>, Tuple<Ts...> t)
{
  return MakeTuple(Get<Ns + 1>(t)...);
}

template <typename... Ts>
auto tail(Tuple<Ts...> t)
{
  return tail_impl(std::make_index_sequence<sizeof...(Ts) - 1u>(),
                   std::forward<Tuple<Ts...>>(t));
}

//======================================================================================//

template <typename List>
class PopFrontT;

template <typename Head, typename... Tail>
class PopFrontT<Tuple<Head, Tail...>> {
public:
  using Type = Tuple<Tail...>;
};

template <typename List>
using PopFront = typename PopFrontT<List>::Type;

//======================================================================================//

template <typename List, typename NewElement>
class PushBackT;

template <typename... Elements, typename NewElement>
class PushBackT<Tuple<Elements...>, NewElement> {
public:
  using Type = Tuple<Elements..., NewElement>;
};

template <typename List, typename NewElement>
using PushBack = typename PushBackT<List, NewElement>::Type;

//================================================================================================//

template <typename _Ret>
struct _ApplyImpl {
  template <typename _Fn, typename _Tuple, size_t... _Idx>
  static _Ret apply_all(_Fn &&__f, _Tuple &&__t, std::index_sequence<_Idx...>)
  {
    return __f(Get<_Idx>(std::forward<_Tuple>(__t))...);
  }
};

//======================================================================================//

template <typename _Tp, typename _Tuple>
struct IndexOf;

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... Types>
struct IndexOf<_Tp, Tuple<_Tp, Types...>> {
  static constexpr std::size_t value = 0;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Head, typename... Tail>
struct IndexOf<_Tp, Tuple<Head, Tail...>> {
  static constexpr std::size_t value = 1 + IndexOf<_Tp, Tuple<Tail...>>::value;
};

//======================================================================================//
// get the index of a type in expansion
//
template <typename _Tp, typename Type>
struct index_of;

template <typename _Tp, typename... Types>
struct index_of<_Tp, std::tuple<_Tp, Types...>> {
  static constexpr std::size_t value = 0;
};

template <typename _Tp, typename Head, typename... Tail>
struct index_of<_Tp, std::tuple<Head, Tail...>> {
  static constexpr std::size_t value = 1 + index_of<_Tp, std::tuple<Tail...>>::value;
};

//================================================================================================//

template <>
struct _ApplyImpl<void> {
  //----------------------------------------------------------------------------------//

  template <std::size_t _N, std::size_t _Nt, typename _Access, typename... _Args,
            std::enable_if_t<(_N == _Nt), int> = 0>
  static void apply(_Args &&... __args)
  {
    // call constructor
    using ApplyType = typename std::tuple_element<_N, _Access>::type;
    ApplyType(std::forward<_Args>(__args)...);
  }

  template <std::size_t _N, std::size_t _Nt, typename _Access, typename... _Args,
            std::enable_if_t<(_N < _Nt), int> = 0>
  static void apply(_Args &&... __args)
  {
    // call constructor
    using ApplyType = typename std::tuple_element<_N, _Access>::type;
    ApplyType(std::forward<_Args>(__args)...);
    // recursive call
    apply<_N + 1, _Nt, _Access, _Args...>(std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <typename _Fn, typename _Tuple, size_t... _Idx>
  static void apply_all(_Fn &&__f, _Tuple &&__t, std::index_sequence<_Idx...>)
  {
    __f(Get<_Idx>(std::forward<_Tuple>(__t))...);
  }

  //--------------------------------------------------------------------------------------------//

  template <typename _Tuple, typename _Obj, typename _Next = PopFront<_Tuple>,
            std::size_t _N    = TupleSize<_Next>::value,
            typename _Indices = std::make_index_sequence<_N>, size_t Idx, size_t... _Idx,
            std::enable_if_t<(_N < 1), int> = 0>
  static void apply_once(_Tuple &&__t, _Obj &&__o, std::index_sequence<Idx, _Idx...>)
  {
    Get<Idx>(__t)(__o);
  }

  template <typename _Tuple, typename _Obj,
            typename _Next    = PopFront<std::decay_t<_Tuple>>,
            std::size_t _N    = TupleSize<_Next>::value,
            typename _Indices = std::make_index_sequence<_N>, size_t Idx, size_t... _Idx,
            std::enable_if_t<(_N > 0), int> = 0>
  static void apply_once(_Tuple &&__t, _Obj &&__o, std::index_sequence<Idx, _Idx...>)
  {
    Get<Idx>(__t)(__o);
    apply_once<_Obj, _Next>(std::forward<_Next>(_Next(Get<_Idx>(__t)...)),
                            std::forward<_Obj>(__o), _Indices{});
  }

  //----------------------------------------------------------------------------------//

  template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
            typename... _Args, std::enable_if_t<(_N == _Nt), int> = 0>
  static void apply_access(_Tuple &&__t, _Args &&... __args)
  {
    // call constructor
    using Type       = decltype(Get<_N>(__t));
    using AccessType = typename TupleElement<_N, _Access>::type;
    AccessType(std::forward<Type>(Get<_N>(__t)), std::forward<_Args>(__args)...);
  }

  template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
            typename... _Args, std::enable_if_t<(_N < _Nt), int> = 0>
  static void apply_access(_Tuple &&__t, _Args &&... __args)
  {
    // call constructor
    using Type       = decltype(Get<_N>(__t));
    using AccessType = typename TupleElement<_N, _Access>::type;
    AccessType(std::forward<Type>(Get<_N>(__t)), std::forward<_Args>(__args)...);
    // recursive call
    apply_access<_N + 1, _Nt, _Access, _Tuple, _Args...>(std::forward<_Tuple>(__t),
                                                         std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename... _Args,
            std::enable_if_t<(_N == _Nt), int> = 0>
  static void apply_loop(_Tuple &&__t, _Args &&... __args)
  {
    // call operator()
    Get<_N>(__t)(std::forward<_Args>(__args)...);
  }

  template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename... _Args,
            std::enable_if_t<(_N < _Nt), int> = 0>
  static void apply_loop(_Tuple &&__t, _Args &&... __args)
  {
    // call operator()
    Get<_N>(__t)(std::forward<_Args>(__args)...);
    // recursive call
    apply_loop<_N + 1, _Nt, _Tuple, _Args...>(std::forward<_Tuple>(__t),
                                              std::forward<_Args>(__args)...);
  }

  //----------------------------------------------------------------------------------//

  template <typename _Tp, typename _Funct, typename... _Args,
            std::enable_if_t<std::is_pointer<_Tp>::value, int> = 0>
  static void apply_function(_Tp &&__t, _Funct &&__f, _Args &&... __args)
  {
    (__t->*__f)(std::forward<_Args>(__args)...);
  }

  //----------------------------------------------------------------------------------//

  template <typename _Tp, typename _Funct, typename... _Args,
            std::enable_if_t<!std::is_pointer<_Tp>::value, int> = 0>
  static void apply_function(_Tp &&__t, _Funct &&__f, _Args &&... __args)
  {
    (__t.*__f)(std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
            typename... _Args, std::enable_if_t<(_N == _Nt), int> = 0>
  static void apply_functions(_Tuple &&__t, _Funct &&__f, _Args &&... __args)
  {
    // call member function at index _N
    using __T = decltype(Get<_N>(__t));
    using __F = decltype(Get<_N>(__f));
    apply_function(std::forward<__T>(Get<_N>(__t)), std::forward<__F>(Get<_N>(__f)),
                   std::forward<_Args>(__args)...);
  }

  template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
            typename... _Args, std::enable_if_t<(_N < _Nt), int> = 0>
  static void apply_functions(_Tuple &&__t, _Funct &&__f, _Args &&... __args)
  {
    // call member function at index _N
    using __T = decltype(Get<_N>(__t));
    using __F = decltype(Get<_N>(__f));
    apply_function(std::forward<__T>(Get<_N>(__t)), std::forward<__F>(Get<_N>(__f)),
                   std::forward<_Args>(__args)...);
    // recursive call
    apply_functions<_N + 1, _Nt, _Tuple, _Funct, _Args...>(
        std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
        std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <std::size_t _N, std::size_t _Nt, typename _Funct, typename... _Args,
            std::enable_if_t<(_N == _Nt), int> = 0>
  static void unroll(_Funct &&__f, _Args &&... __args)
  {
    (__f)(std::forward<_Args>(__args)...);
  }

  template <std::size_t _N, std::size_t _Nt, typename _Funct, typename... _Args,
            std::enable_if_t<(_N < _Nt), int> = 0>
  static void unroll(_Funct &&__f, _Args &&... __args)
  {
    (__f)(std::forward<_Args>(__args)...);
    unroll<_N + 1, _Nt, _Funct, _Args...>(std::forward<_Funct>(__f),
                                          std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
            typename... _Args, std::enable_if_t<(_N == _Nt), int> = 0>
  static void unroll_members(_Tuple &&__t, _Funct &&__f, _Args &&... __args)
  {
    (Get<_N>(__t)).*(Get<_N>(__f))(std::forward<_Args>(__args)...);
  }

  template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
            typename... _Args, std::enable_if_t<(_N < _Nt), int> = 0>
  static void unroll_members(_Tuple &&__t, _Funct &&__f, _Args &&... __args)
  {
    (Get<_N>(__t)).*(Get<_N>(__f))(std::forward<_Args>(__args)...);
    unroll_members<_N + 1, _Nt, _Tuple, _Funct, _Args...>(std::forward<_Tuple>(__t),
                                                          std::forward<_Funct>(__f),
                                                          std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <std::size_t _N, std::size_t _Nt, typename _Access, typename... _Args,
            std::enable_if_t<(_N == _Nt), char> = 0>
  static void unroll_indices(_Args &&... __args)
  {
    // call constructor
    using AccessType = typename std::tuple_element<_N, _Access>::type;
    AccessType(_N, std::forward<_Args>(__args)...);
  }

  template <std::size_t _N, std::size_t _Nt, typename _Access, typename... _Args,
            std::enable_if_t<(_N < _Nt), char> = 0>
  static void unroll_indices(_Args &&... __args)
  {
    // call constructor
    using AccessType = typename std::tuple_element<_N, _Access>::type;
    AccessType(_N, std::forward<_Args>(__args)...);
    // recursive call
    unroll_indices<_N + 1, _Nt, _Access, _Args...>(std::forward<_Args>(__args)...);
  }

  //----------------------------------------------------------------------------------//
};

//================================================================================================//

template <typename _Ret>
struct Apply {
  template <typename _Fn, typename _Tuple,
            std::size_t _N    = TupleSize<std::decay_t<_Tuple>>::value,
            typename _Indices = std::make_index_sequence<_N>>
  static _Ret apply_all(_Fn &&__f, _Tuple &&__t)
  {
    return _ApplyImpl<_Ret>::template apply_all<_Fn, _Tuple>(
        std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
  }
};

//================================================================================================//

template <>
struct Apply<void> {
  //----------------------------------------------------------------------------------//

  template <typename _Access, typename... _Args,
            size_t _N = std::tuple_size<_Access>::value>
  static void apply(_Args &&... __args)
  {
    _ApplyImpl<void>::template apply<0, _N - 1, _Access, _Args...>(
        std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <typename _Fn, typename _Tuple,
            std::size_t _N    = TupleSize<std::decay_t<_Tuple>>::value,
            typename _Indices = std::make_index_sequence<_N>>
  static void apply_all(_Fn &&__f, _Tuple &&__t)
  {
    _ApplyImpl<void>::template apply_all<_Fn, _Tuple>(
        std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
  }

  //--------------------------------------------------------------------------------------------//

  template <typename _Tuple, typename _Obj,
            std::size_t _N    = TupleSize<std::decay_t<_Tuple>>::value,
            typename _Indices = std::make_index_sequence<_N>>
  static void apply_once(_Tuple &&__t, _Obj &&__o)
  {
    _ApplyImpl<void>::template apply_once<_Tuple, _Obj>(
        std::forward<_Tuple>(__t), std::forward<_Obj>(__o), _Indices{});
  }

  //----------------------------------------------------------------------------------//

  template <typename _Access, typename _Tuple, typename... _Args,
            std::size_t _N = TupleSize<std::decay_t<_Tuple>>::value>
  static void apply_access(_Tuple &__t, _Args &&... __args)
  {
    _ApplyImpl<void>::template apply_access<0, _N - 1, _Access, _Tuple, _Args...>(
        std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
  }

  //----------------------------------------------------------------------------------//

  template <typename _Access, typename _Tuple, typename... _Args,
            std::size_t _N = TupleSize<std::decay_t<_Tuple>>::value>
  static void apply_access(_Tuple &&__t, _Args &&... __args)
  {
    _ApplyImpl<void>::template apply_access<0, _N - 1, _Access, _Tuple, _Args...>(
        std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
  }

  //----------------------------------------------------------------------------------//

  template <typename _Access, typename _Tuple,
            std::size_t _N = TupleSize<std::decay_t<_Tuple>>::value>
  static void apply_access(_Tuple &&__t)
  {
    _ApplyImpl<void>::template apply_access<0, _N - 1, _Access, _Tuple>(
        std::forward<_Tuple>(__t));
  }

  //--------------------------------------------------------------------------------------------//

  template <typename _Tuple, typename... _Args,
            std::size_t _N = TupleSize<std::decay_t<_Tuple>>::value>
  static void apply_loop(_Tuple &&__t, _Args &&... __args)
  {
    _ApplyImpl<void>::template apply_loop<0, _N - 1, _Tuple, _Args...>(
        std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <typename _Tuple, typename _Funct, typename... _Args,
            std::size_t _Nt = TupleSize<std::decay_t<_Tuple>>::value,
            std::size_t _Nf = TupleSize<std::decay_t<_Funct>>::value>
  static void apply_functions(_Tuple &&__t, _Funct &&__f, _Args &&... __args)
  {
    static_assert(_Nt == _Nf, "tuple_size of objects must match tuple_size of functions");
    _ApplyImpl<void>::template apply_functions<0, _Nt - 1, _Tuple, _Funct, _Args...>(
        std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
        std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <std::size_t _N, typename _Func, typename... _Args>
  static void unroll(_Func &&__f, _Args &&... __args)
  {
    _ApplyImpl<void>::template unroll<0, _N - 1, _Func, _Args...>(
        std::forward<_Func>(__f), std::forward<_Args>(__args)...);
  }

  //--------------------------------------------------------------------------------------------//

  template <std::size_t _N, typename _Tuple, typename _Func, typename... _Args>
  static void unroll_members(_Tuple &&__t, _Func &&__f, _Args &&... __args)
  {
    _ApplyImpl<void>::template unroll_members<0, _N - 1, _Tuple, _Func, _Args...>(
        std::forward<_Tuple>(__t), std::forward<_Func>(__f),
        std::forward<_Args>(__args)...);
  }

  //----------------------------------------------------------------------------------//

  template <typename _Access, typename... _Args,
            std::size_t _N = std::tuple_size<std::decay_t<_Access>>::value,
            std::enable_if_t<(_N > 0), int> = 0>
  static void unroll_indices(_Args &&... _args)
  {
    _ApplyImpl<void>::template unroll_indices<0, _N - 1, _Access, _Args...>(
        std::forward<_Args>(_args)...);
  }

  //--------------------------------------------------------------------------------------------//
};

//======================================================================================//
// check if type is in expansion
//
namespace impl {
template <typename...>
struct is_one_of {
  static constexpr bool value = false;
};

template <typename F, typename S, template <typename...> class _Tuple, typename... T>
struct is_one_of<F, S, _Tuple<T...>> {
  static constexpr bool value =
      (std::is_same<F, S>::value || is_one_of<F, _Tuple<T...>>::value);
};

template <typename F, typename S, template <typename...> class _Tuple, typename... T>
struct is_one_of<F, _Tuple<S, T...>> {
  static constexpr bool value = is_one_of<F, S, _Tuple<T...>>::value;
};

} // namespace impl

template <typename _Tp, typename _Types>
using is_one_of = typename impl::is_one_of<_Tp, _Types>;

//================================================================================================//
