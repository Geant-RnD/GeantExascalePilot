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
 * @brief Tasking helper routine.
 */
//===----------------------------------------------------------------------===//

#pragma once

//============================================================================//

#include "Geant/core/Config.hpp"
#include "Geant/core/Logger.hpp"

#include "PTL/TBBTaskGroup.hh"
#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"

namespace geantx {
// make these available in global namespace
using PTL::ConsumeParameters;
using PTL::GetEnv;

namespace tasking {
using namespace PTL;

inline void init_thread_data(ThreadPool *tp)
{
  ThreadData *&thread_data = ThreadData::GetInstance();
  if (!thread_data) thread_data = new ThreadData(tp);
  thread_data->is_master   = false;
  thread_data->within_task = false;
}

//======================================================================================//

inline std::unique_ptr<TaskRunManager> &cpu_run_manager()
{
  typedef std::unique_ptr<TaskRunManager> pointer;
  static thread_local pointer _instance = pointer(
      new TaskRunManager(GetEnv<bool>("GEANT_USE_TBB", false, "Enable TBB backend")));
  return _instance;
}

//======================================================================================//

inline std::unique_ptr<TaskRunManager> &gpu_run_manager()
{
  typedef std::unique_ptr<TaskRunManager> pointer;
  static thread_local pointer _instance = pointer(
      new TaskRunManager(GetEnv<bool>("GEANT_USE_TBB", false, "Enable TBB backend")));
  return _instance;
}

//======================================================================================//

inline void init_run_manager(std::unique_ptr<TaskRunManager> &run_man,
                             uintmax_t nthreads = GetEnv<uintmax_t>(
                                 "GEANT_NUM_THREADS", std::thread::hardware_concurrency(),
                                 "Number of threads"))
{
  // register thread-id
  auto tid = ThreadPool::GetThisThreadID();
  ConsumeParameters(tid);
  // get verbosity
  static int verbose = GetEnv<int>("TASKING_VERBOSE", 0);

  {
    // instance should be thread-local but just in-case
    AutoLock lman(TypeMutex<TaskRunManager>());
    if (!run_man->IsInitialized()) {
      if (verbose > 0) {
        geantx::Log(geantx::kStatus)
            << "\n"
            << "[" << tid << "] Initializing tasking run manager with " << nthreads
            << " threads...";
      }
      run_man->Initialize(nthreads);
    }
  }
  init_thread_data(run_man->GetThreadPool());
}

} // namespace tasking
} // namespace geantx
