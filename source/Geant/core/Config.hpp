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
#error "GeantX requires C++14"
#endif

// Include global definitions from VecCore and VecGeom
#include "base/Global.h"

#if !defined(GEANT_CUDA) && defined(VECCORE_CUDA)
#define GEANT_CUDA
#endif

#if !defined(GEANT_CUDA_DEVICE_COMPILATION) && defined(VECCORE_CUDA_DEVICE_COMPILATION)
#define GEANT_CUDA_DEVICE_COMPILATION
#endif

// Inlining
#ifdef __INTEL_COMPILER
#define GEANT_FORCE_INLINE inline
#else
#if (defined(__GNUC__) || defined(__GNUG__)) && !defined(__clang__) && \
    !defined(__NO_INLINE__) && !defined(GEANT_NOINLINE)
#define GEANT_FORCE_INLINE inline __attribute__((always_inline))
#else
// Clang or forced inlining is disabled ( by falling back to compiler decision )
#define GEANT_FORCE_INLINE inline
#endif
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

#define GEANT_IMPL_NAMESPACE cxx

#define GEANT_DECLARE_CONSTANT(type, name) \
  namespace host_constant {                \
  extern const type name;                  \
  }                                        \
  using host_constant::name
#else

#define GEANT_IMPL_NAMESPACE cuda

#ifdef GEANT_CUDA_DEVICE_COMPILATION
#ifdef CUDA_SEP_COMP
#define GEANT_DECLARE_CONSTANT(type, name) \
  namespace host_constant {                \
  extern const type name;                  \
  }                                        \
  namespace device_constant {              \
  extern __constant__ type name;           \
  }                                        \
  using device_constant::name
#else // CUDA_SEP_COMP
#define GEANT_DECLARE_CONSTANT(type, name) \
  namespace host_constant {                \
  extern const type name;                  \
  }                                        \
  namespace device_constant {              \
  __constant__ type name;                  \
  }                                        \
  using device_constant::name
#endif // CUDA_SEP_COMP
#else  // GEANT_CUDA_DEVICE_COMPILATION
#ifdef CUDA_SEP_COMP
#define GEANT_DECLARE_CONSTANT(type, name) \
  namespace host_constant {                \
  extern const type name;                  \
  }                                        \
  namespace device_constant {              \
  extern __constant__ type name;           \
  }                                        \
  using host_constant::name
#else // CUDA_SEP_COMP
#define GEANT_DECLARE_CONSTANT(type, name) \
  namespace host_constant {                \
  extern const type name;                  \
  }                                        \
  namespace device_constant {              \
  __constant__ type name;                  \
  }                                        \
  using host_constant::name
#endif // CUDA_SEP_COMP
#endif // Device build or not
#endif // gcc or nvcc

#ifndef GEANT_CUDA

#define GEANT_DEVICE_DECLARE_CONV(NS, classOrStruct, X) \
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

#define GEANT_DEVICE_DECLARE_CONV(NS, classOrStruct, X) class __QuietSemi

#endif

//======================================================================================//

#if !defined(PRAGMA_SIMD)
#define PRAGMA_SIMD _Pragma("omp simd")
#endif

//======================================================================================//

#if defined(__NVCC__)
#define GEANT_HOST __host__
#define GEANT_DEVICE __device__
#define GEANT_HOST_DEVICE __host__ __device__
#define GEANT_GLOBAL __global__
#else
#define GEANT_HOST
#define GEANT_DEVICE
#define GEANT_HOST_DEVICE
#define GEANT_GLOBAL
#endif

//======================================================================================//
//
//      Operating System
//
//======================================================================================//

// machine bits
#if defined(__x86_64__)
#if !defined(_64BIT)
#define _64BIT
#endif
#else
#if !defined(_32BIT)
#define _32BIT
#endif
#endif

//--------------------------------------------------------------------------------------//
// base operating system

#if defined(_WIN32) || defined(_WIN64)
#if !defined(_WINDOWS)
#define _WINDOWS
#endif
//--------------------------------------------------------------------------------------//

#elif defined(__APPLE__) || defined(__MACH__)
#if !defined(_MACOS)
#define _MACOS
#endif
#if !defined(_UNIX)
#define _UNIX
#endif
//--------------------------------------------------------------------------------------//

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#if !defined(_LINUX)
#define _LINUX
#endif
#if !defined(_UNIX)
#define _UNIX
#endif
//--------------------------------------------------------------------------------------//

#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(_)
#if !defined(_UNIX)
#define _UNIX
#endif
#endif

//--------------------------------------------------------------------------------------//

#if defined(_LINUX) || defined(_MACOS)
#define _C_UNIX // common unix derivative (i.e. Linux or macOS)
#endif

//======================================================================================//

