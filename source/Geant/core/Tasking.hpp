//
// created by jrmadsen on Fri Mar  8 13:30:31 2019
//

#pragma once

//============================================================================//

#include "PTL/TBBTaskGroup.hh"
#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/TaskManager.hh"
#include "PTL/TaskRunManager.hh"
#include "PTL/ThreadData.hh"
#include "PTL/ThreadPool.hh"
#include "PTL/Threading.hh"

namespace geant {
inline namespace tasking {
inline void init_thread_data(ThreadPool *tp)
{
  ThreadData *&thread_data = ThreadData::GetInstance();
  if (!thread_data) thread_data = new ThreadData(tp);
  thread_data->is_master   = false;
  thread_data->within_task = false;
}

//======================================================================================//

inline TaskRunManager *cpu_run_manager()
{
  typedef std::shared_ptr<TaskRunManager> pointer;
  static thread_local pointer _instance =
      pointer(new TaskRunManager(GetEnv<bool>("GEANT_USE_TBB", false, "Enable TBB backend")));
  return _instance.get();
}

//======================================================================================//

inline TaskRunManager *gpu_run_manager()
{
  typedef std::shared_ptr<TaskRunManager> pointer;
  static thread_local pointer _instance =
      pointer(new TaskRunManager(GetEnv<bool>("GEANT_USE_TBB", false, "Enable TBB backend")));
  return _instance.get();
}

//======================================================================================//

inline void init_run_manager(TaskRunManager *run_man,
                             uintmax_t nthreads = GetEnv<uintmax_t>("GEANT_NUM_THREADS",
                                                                    std::thread::hardware_concurrency(),
                                                                    "Number of threads"))
{
  // register thread-id
  auto tid = ThreadPool::GetThisThreadID();
  ConsumeParameters(tid);
  // get verbosity
  static int verbose = GetEnv<int>("TASKING_VERBOSE", 0);

  {
    // instance should be thread-local but just in-case
    AutoLock l(TypeMutex<TaskRunManager>());
    if (!run_man->IsInitialized()) {
      if (verbose > 0) {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cout << "\n"
                  << "[" << tid << "] Initializing tasking run manager with " << nthreads << " threads..." << std::endl;
      }
      run_man->Initialize(nthreads);
    }
  }
  init_thread_data(run_man->GetThreadPool());
}

} // namespace tasking

} // namespace geant
