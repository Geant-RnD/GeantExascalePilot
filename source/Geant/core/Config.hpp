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
 * @brief GeantX configuration macros
 */
//===----------------------------------------------------------------------===//

#pragma once

#if __cplusplus < 201103L && !defined(__NVCC__)
#  error "GeantX requires C++14"
#endif

// Include global definitions from VecCore and VecGeom
#include "VecGeom/base/Global.h"

#if !defined(GEANT_CUDA) && defined(VECCORE_CUDA)
#  define GEANT_CUDA
#endif

#if !defined(GEANT_CUDA_DEVICE_COMPILATION) && defined(VECCORE_CUDA_DEVICE_COMPILATION)
#  define GEANT_CUDA_DEVICE_COMPILATION
#endif

// Inlining
#ifdef __INTEL_COMPILER
#  define GEANT_FORCE_INLINE inline
#else
#  if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__clang__) && \
      !defined(__NO_INLINE__) && !defined(GEANT_NOINLINE)
#    define GEANT_FORCE_INLINE inline __attribute__((always_inline))
#  else
// Clang or forced inlining is disabled ( by falling back to compiler decision )
#    define GEANT_FORCE_INLINE inline
#  endif
#endif

// Utility macros
#define GEANT_BIT(n) (1ULL << (n))
#define GEANT_SETBIT(n, i) ((n) |= BIT(i))
#define GEANT_CLRBIT(n, i) ((n) &= ~BIT(i))
#define GEANT_TESTBIT(n, i) ((bool)(((n)&BIT(i)) != 0))

//////////////////////////////////////////
// Declaration for constant define in the
// device constant section. Use:
//    GEANT_DECLARE_CONST(double,gTolerance);
//
// This will declare the following:
//    extern double host_constant::gTolerance;
// and only in nvcc
//    __constant device_constant::gTolerance;
//
// In gcc and nvcc host code host_constant::gTolerance is aliased
// to geantx::cxx::gTolerance and geantx::cuda::gTolerance respectively.
// In nvcc device code, geantx::cuda::gTolerance is aliased to
// device_constant::gTolerance.

#ifndef GEANT_CUDA

#  define GEANT_IMPL_NAMESPACE cxx

#  define GEANT_DECLARE_CONSTANT(type, name) \
    namespace host_constant {                \
    extern const type name;                  \
    }                                        \
    using host_constant::name
#else

#  define GEANT_IMPL_NAMESPACE cuda

#  ifdef GEANT_CUDA_DEVICE_COMPILATION
#    ifdef CUDA_SEP_COMP
#      define GEANT_DECLARE_CONSTANT(type, name) \
        namespace host_constant {                \
        extern const type name;                  \
        }                                        \
        namespace device_constant {              \
        extern __constant__ type name;           \
        }                                        \
        using device_constant::name
#    else // CUDA_SEP_COMP
#      define GEANT_DECLARE_CONSTANT(type, name) \
        namespace host_constant {                \
        extern const type name;                  \
        }                                        \
        namespace device_constant {              \
        __constant__ type name;                  \
        }                                        \
        using device_constant::name
#    endif // CUDA_SEP_COMP
#  else    // GEANT_CUDA_DEVICE_COMPILATION
#    ifdef CUDA_SEP_COMP
#      define GEANT_DECLARE_CONSTANT(type, name) \
        namespace host_constant {                \
        extern const type name;                  \
        }                                        \
        namespace device_constant {              \
        extern __constant__ type name;           \
        }                                        \
        using host_constant::name
#    else // CUDA_SEP_COMP
#      define GEANT_DECLARE_CONSTANT(type, name) \
        namespace host_constant {                \
        extern const type name;                  \
        }                                        \
        namespace device_constant {              \
        __constant__ type name;                  \
        }                                        \
        using host_constant::name
#    endif // CUDA_SEP_COMP
#  endif   // Device build or not
#endif     // gcc or nvcc

#ifndef GEANT_CUDA

#  define GEANT_DEVICE_DECLARE_CONV(NS, classOrStruct, X) \
    namespace NS {                                        \
    namespace cuda {                                      \
    classOrStruct X;                                      \
    }                                                     \
    inline namespace cxx {                                \
    classOrStruct X;                                      \
    }                                                     \
    }                                                     \
    namespace vecgeom {                                   \
    template <>                                           \
    struct kCudaType<NS::cxx::X> {                        \
      using type_t = NS::cuda::X;                         \
    };                                                    \
    }                                                     \
    class __QuietSemi

#else

#  define GEANT_DEVICE_DECLARE_CONV(NS, classOrStruct, X) class __QuietSemi

#endif

//======================================================================================//

#if defined(__NVCC__)
#  if !defined(GEANT_HOST)
#    define GEANT_HOST __host__
#  endif

#  if !defined(GEANT_DEVICE)
#    define GEANT_DEVICE __device__
#  endif

#  if !defined(GEANT_HOST_DEVICE)
#    define GEANT_HOST_DEVICE __host__ __device__
#  endif

#  if !defined(GEANT_GLOBAL)
#    define GEANT_GLOBAL __global__
#  endif

#else

#  if !defined(GEANT_HOST)
#    define GEANT_HOST
#  endif

#  if !defined(GEANT_DEVICE)
#    define GEANT_DEVICE
#  endif

#  if !defined(GEANT_HOST_DEVICE)
#    define GEANT_HOST_DEVICE
#  endif

#  if !defined(GEANT_GLOBAL)
#    define GEANT_GLOBAL
#  endif
#endif

//======================================================================================//

/*---- unlikely / likely expressions -----------------------------------------*/
// These are meant to use in cases like:
//   if (R__unlikely(expression)) { ... }
// in performance-critical sessions.  R__unlikely / R__likely provide hints to
// the compiler code generation to heavily optimize one side of a conditional,
// causing the other branch to have a heavy performance cost.
//
// It is best to use this for conditionals that test for rare error cases or
// backward compatibility code.

#if (__GNUC__ >= 3) || defined(__INTEL_COMPILER)
#  if !defined(GEANT_UNLIKELY)
#    define GEANT_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#  endif
#  if !defined(GEANT_LIKELY)
#    define GEANT_LIKELY(expr) __builtin_expect(!!(expr), 1)
#  endif
#else
#  define GEANT_UNLIKELY(expr) expr
#  define GEANT_LIKELY(expr) expr
#endif

//======================================================================================//

// @brief Assertion routines.

// Contract validation on input
#define REQUIRE(code)

// Internal consistency checking
#define CHECK(code)

// Contract validation on output
#define ENSURE(code)

// Always-on assertion that prints message if it fails
#define INSIST(code, msg_stream)
