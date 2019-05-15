// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

//======================================================================================//
//  C headers

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

//======================================================================================//
//  C++ headers

#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Profiler.hpp"

#include "PTL/AutoLock.hh"
#include "PTL/TBBTaskGroup.hh"
#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"
#include "PTL/Types.hh"
#include "PTL/Utility.hh"

//--------------------------------------------------------------------------------------//

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

//======================================================================================//
// this function is used by a macro -- returns a unique identifier to the thread
inline uintmax_t GetThisThreadID()
{
  return ThreadPool::GetThisThreadID();
}

//======================================================================================//

#if !defined(scast)
#define scast static_cast
#endif

//======================================================================================//

#if !defined(HW_CONCURRENCY)
#define HW_CONCURRENCY std::thread::hardware_concurrency()
#endif

//======================================================================================//

#if !defined(_Forward_t)
#define _Forward_t(TYPE, VAL) std::forward<TYPE>(VAL)...
#endif

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
