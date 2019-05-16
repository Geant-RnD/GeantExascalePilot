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
 * @brief Utility macros
 */
//===----------------------------------------------------------------------===//

#pragma once

//======================================================================================//
//  C headers
//
#include <cassert>
#include <cstdint>
#include <cstdlib>

//======================================================================================//
//  C++ headers
//
#include <type_traits>
#include <chrono>
#include <sstream>

//--------------------------------------------------------------------------------------//
// used for ThreadPool::GetThreadID()
//
#include "PTL/ThreadPool.hh"

//--------------------------------------------------------------------------------------//
// cuda headers for debug macros
//
#include <cuda.h>
#include <cuda_runtime_api.h>

//======================================================================================//
// this function is used by a macro -- returns a unique identifier to the thread
inline uintmax_t GetThisThreadID()
{
  return PTL::ThreadPool::GetThisThreadID();
}
//======================================================================================//

#if !defined(PRAGMA_SIMD)
#define PRAGMA_SIMD _Pragma("omp simd")
#endif

//======================================================================================//

#if !defined(PRINT_HERE)
#define PRINT_HERE(extra)                                                               \
  printf("[%lu]> %s@'%s':%i %s\n", GetThisThreadID(), __FUNCTION__, __FILE__, __LINE__, \
         extra)
#endif

//======================================================================================//

#if !defined(START_TIMER)
#define START_TIMER(var) auto var = std::chrono::system_clock::now()
#endif

//======================================================================================//
// macro for recording a time point
#if !defined(GET_TIMER)
#define GET_TIMER(var) auto var = std::chrono::high_resolution_clock::now()
#endif

//======================================================================================//

#if !defined(REPORT_TIMER)
#define REPORT_TIMER(start_time, note, counter, total_count)                          \
  {                                                                                   \
    auto end_time                                 = std::chrono::system_clock::now(); \
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;            \
    printf("[%li]> %-16s :: %3i of %3i... %5.2f seconds\n", GetThisThreadID(), note,  \
           counter, total_count, elapsed_seconds.count());                            \
  }
#endif

//======================================================================================//

#if !defined(CUDA_CHECK_LAST_ERROR)
#define CUDA_CHECK_LAST_ERROR()                                               \
  {                                                                           \
    cudaStreamSynchronize(0);                                                 \
    cudaError err = cudaGetLastError();                                       \
    if (cudaSuccess != err) {                                                 \
      std::stringstream ss;                                                   \
      ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__ \
         << "':" << __LINE__ << " : " << cudaGetErrorString(err);             \
      fprintf(stderr, "%s\n", ss.str().c_str());                              \
      throw std::runtime_error(ss.str().c_str());                             \
    }                                                                         \
  }
#endif

//======================================================================================//

#if !defined(CUDA_DEBUG_LAST_ERROR)
#if defined(DEBUG)
#define CUDA_DEBUG_LAST_ERROR()                                               \
  {                                                                           \
    cudaStreamSynchronize(0);                                                 \
    cudaError err = cudaGetLastError();                                       \
    if (cudaSuccess != err) {                                                 \
      std::stringstream ss;                                                   \
      ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__ \
         << "':" << __LINE__ << " : " << cudaGetErrorString(err);             \
      fprintf(stderr, "%s\n", ss.str().c_str());                              \
      throw std::runtime_error(ss.str().c_str());                             \
    }                                                                         \
  }
#else
#define CUDA_DEBUG_LAST_ERROR() \
  {                             \
    ;                           \
  }
#endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(CUDA_CHECK_CALL)
#define CUDA_CHECK_CALL(err)                                                  \
  {                                                                           \
    if (cudaSuccess != err) {                                                 \
      std::stringstream ss;                                                   \
      ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__ \
         << "':" << __LINE__ << " : " << cudaGetErrorString(err);             \
      fprintf(stderr, "%s\n", ss.str().c_str());                              \
      throw std::runtime_error(ss.str().c_str());                             \
    }                                                                         \
  }
#endif

//======================================================================================//
//
//      NVTX macros
//
//======================================================================================//

#if defined(GEANT_USE_NVTX)
#include <nvToolsExt.h>

#ifndef NVTX_RANGE_PUSH
#define NVTX_RANGE_PUSH(obj) nvtxRangePushEx(obj)
#endif
#ifndef NVTX_RANGE_POP
#define NVTX_RANGE_POP(obj)   \
  cudaStreamSynchronize(obj); \
  nvtxRangePop()
#endif
#ifndef NVTX_NAME_THREAD
#define NVTX_NAME_THREAD(num, name) nvtxNameOsThread(num, name)
#endif
#ifndef NVTX_MARK
#define NVTX_MARK(msg) nvtxMark(name)
#endif
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
