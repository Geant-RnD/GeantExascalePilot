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
 * @brief Profiler interface(s)
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <timemory/timemory.hpp>

#include "Geant/core/Logger.hpp"

//--------------------------------------------------------------------------------------//
// used for ThreadPool::GetThreadID()
//
#include "PTL/ThreadPool.hh"

//======================================================================================//
// macro for recording a time point
#if !defined(GEANT_GET_TIMER)
#  define GEANT_GET_TIMER(var) auto var = std::chrono::high_resolution_clock::now()
#endif

//======================================================================================//

#if !defined(GEANT_REPORT_TIMER)
// Format string below is: "[%li]> %-16s :: %3i of %3i... %5.2f seconds\n"
#  define GEANT_REPORT_TIMER(start_time, note, counter, total_count)                  \
    {                                                                                 \
      auto end_time = std::chrono::high_resolution_clock::now();                      \
                                                                                      \
      std::chrono::duration<double> elapsed_seconds = end_time - start_time;          \
      geantx::Log(kInfo) << "[" << ThreadPool::GetThisThreadID() << "]> "             \
                         << std::setw(16) << std::setiosflags(ios::left) << note      \
                         << " :: " << std::setiosflags(ios::right) << std::setw(3)    \
                         << counter << " of " << total_count << "..." << std::setw(5) \
                         << std::setprecision(2) << elapsed_seconds.count()           \
                         << "seconds ";                                               \
    }
#endif

//======================================================================================//
//
//      NVTX macros
//
//======================================================================================//

#if defined(GEANT_USE_NVTX)
#  include <nvToolsExt.h>

#  ifndef NVTX_RANGE_PUSH
#    define NVTX_RANGE_PUSH(obj) nvtxRangePushEx(obj)
#  endif
#  ifndef NVTX_RANGE_POP
#    define NVTX_RANGE_POP(obj)   \
      cudaStreamSynchronize(obj); \
      nvtxRangePop()
#  endif
#  ifndef NVTX_NAME_THREAD
#    define NVTX_NAME_THREAD(num, name) nvtxNameOsThread(num, name)
#  endif
#  ifndef NVTX_MARK
#    define NVTX_MARK(msg) nvtxMark(name)
#  endif
#endif

// eventually this will be hidden so that only available to tests
// #if defined(GEANT_TESTING)

//======================================================================================//
//
//      Testing Types
//
//======================================================================================//

using namespace tim::component;
using toolset_t = tim::auto_tuple<real_clock, peak_rss>;
// using toolset_t =
//    tim::auto_tuple<real_clock, system_clock, user_clock, cpu_util, peak_rss>;
// using toolset_t = tim::auto_timer;

// #endif

#define GEANT_THIS_TYPE_TESTING_MARKER(...)                                         \
  TIMEMORY_BLANK_MARKER(toolset_t, tim::demangle(typeid(this_type).name()).c_str(), \
                        "::", __FUNCTION__, "()", __VA_ARGS__);
