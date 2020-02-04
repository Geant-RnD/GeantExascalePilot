//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/processes/ProcessConcepts.hpp
 * @brief Memory pool for device and host allocations.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Logger.hpp"
#include "Geant/core/Profiler.hpp"
#include "Geant/core/Tuple.hpp"

#include <functional>
#include <string>
#include <type_traits>

namespace geantx {
struct Process;
namespace mpl {
//--------------------------------------------------------------------------------------//
//
//                              AT-REST
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType, typename ParticleType>
struct AtRest {
  static_assert(std::is_base_of<Process, ProcessType>::value,
                "ProcessType must derive for Process");

  //----------------------------------------------------------------------------------//
  //                                  GPIL
  //----------------------------------------------------------------------------------//

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
  static double GPIL(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "at rest step generic";
    ProcessType p;
    return p.AtRestGPIL(std::forward<_Args>(args)...);
  }

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
  static double GPIL(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "at rest step specific";
    ProcessType p;
    return p.template AtRestGPIL<ParticleType>(std::forward<_Args>(args)...);
  }

  //----------------------------------------------------------------------------------//
  //                                  DO-IT
  //----------------------------------------------------------------------------------//

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
  static void DoIt(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "at rest step generic";
    ProcessType p;
    p.AtRestDoIt(std::forward<_Args>(args)...);
  }

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
  static void DoIt(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "at rest step specific";
    ProcessType p;
    p.template AtRestDoIt<ParticleType>(std::forward<_Args>(args)...);
  }
};

//--------------------------------------------------------------------------------------//
//
//                              Along-Step
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType, typename ParticleType>
struct AlongStep {
  static_assert(std::is_base_of<Process, ProcessType>::value,
                "ProcessType must derive for Process");

  //----------------------------------------------------------------------------------//
  //                                  GPIL
  //----------------------------------------------------------------------------------//

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
  static double GPIL(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "along step generic";
    ProcessType p;
    return p.AlongStepGPIL(std::forward<_Args>(args)...);
  }

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
  static double GPIL(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "along step specific";
    ProcessType p;
    return p.template AlongStepGPIL<ParticleType>(std::forward<_Args>(args)...);
  }

  //----------------------------------------------------------------------------------//
  //                                  DO-IT
  //----------------------------------------------------------------------------------//

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
  static void DoIt(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "along step generic";
    ProcessType p;
    p.AlongStepDoIt(std::forward<_Args>(args)...);
  }

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
  static void DoIt(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "along step specific";
    ProcessType p;
    p.template AlongStepDoIt<ParticleType>(std::forward<_Args>(args)...);
  }
};

//--------------------------------------------------------------------------------------//
//
//                              Post-Step
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType, typename ParticleType>
struct PostStep {
  static_assert(std::is_base_of<Process, ProcessType>::value,
                "ProcessType must derive for Process");

  //----------------------------------------------------------------------------------//
  //                                  GPIL
  //----------------------------------------------------------------------------------//

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
  static double GPIL(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "post step generic";
    ProcessType p;
    return p.PostStepGPIL(std::forward<_Args>(args)...);
  }

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
  static double GPIL(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "post step specific";
    ProcessType p;
    return p.template PostStepGPIL<ParticleType>(std::forward<_Args>(args)...);
  }

  //----------------------------------------------------------------------------------//
  //                                  DO-IT
  //----------------------------------------------------------------------------------//

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<!(is_one_of<_Part, Specialized>::value), int> = 0>
  static void DoIt(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "post step generic";
    ProcessType p;
    p.PostStepDoIt(std::forward<_Args>(args)...);
  }

  template <typename _Proc = ProcessType, typename _Part = ParticleType,
            typename... _Args, typename Specialized = typename _Proc::specialized_types,
            std::enable_if_t<(is_one_of<_Part, Specialized>::value), int> = 0>
  static void DoIt(_Args &&... args)
  {
    // geantx::Log(kInfo) << GEANT_HERE << "post step specific";
    ProcessType p;
    p.template PostStepDoIt<ParticleType>(std::forward<_Args>(args)...);
  }
};

//--------------------------------------------------------------------------------------//
} // namespace mpl

//======================================================================================//

template <typename ProcessType, typename ParticleType>
struct AtRest {
  using this_type = AtRest<ProcessType, ParticleType>;
  using mpl_type  = mpl::AtRest<ProcessType, ParticleType>;

  //
  //  Invoked for selection among processes that propose a PIL
  //
  template <typename _Track, typename _Tp, typename _Func, typename _Proc = ProcessType,
            std::enable_if_t<(_Proc::EnableAtRestGPIL), int> = 0>
  AtRest(size_t _N, _Track *_track, intmax_t *_doit_idx, _Tp *_doit_value,
         _Func *_doit_apply)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _value = mpl_type::GPIL(_track);
    if (_value < *_doit_value) {
      *_doit_idx   = _N;
      *_doit_value = _value;
      *_doit_apply = [&]() { mpl_type::DoIt(_track); };
    }
  }

  //
  //  Invoked when a process does not provide a PIL proposal.
  //  Compiler will completely eliminate this "function call" in the binary
  //
  template <typename _Track, typename _Tp, typename _Func, typename _Proc = ProcessType,
            std::enable_if_t<!(_Proc::EnableAtRestGPIL), int> = 0>
  AtRest(size_t, _Track *, intmax_t *, _Tp *, _Func *)
  {}

  //
  //  Invoked for selection among processes that DO NOT propose a PIL
  //  but DO have something to do (i.e. something that should always be done)
  //
  template <
      typename _Track, typename _Proc = ProcessType,
      std::enable_if_t<(!_Proc::EnableAtRestGPIL && _Proc::EnableAtRestDoIt), int> = 0>
  explicit AtRest(_Track *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    geantx::Log(kInfo) << GEANT_HERE << "[NO GPIL AT-REST DO-IT] "
                       << tim::demangle<_Proc>();
    mpl_type::DoIt(_track);
  }

  //
  //  Invoked for anything that doesn't fit the conditions above.
  //  Compiler will completely eliminate this "function call" in the binary
  //
  template <
      typename _Track, typename _Proc = ProcessType,
      std::enable_if_t<!(!_Proc::EnableAtRestGPIL && _Proc::EnableAtRestDoIt), int> = 0>
  explicit AtRest(_Track *)
  {}
};

//======================================================================================//

template <typename ProcessType, typename ParticleType>
struct AlongStep {
  using this_type                  = AlongStep<ProcessType, ParticleType>;
  using mpl_type                   = mpl::AlongStep<ProcessType, ParticleType>;
  static constexpr bool EnableDoIt = ProcessType::EnableAlongStepDoIt;

  //
  //  Invoked for selection among processes that propose a PIL
  //
  template <typename _Track, typename _Tp, typename _Func, typename _Proc = ProcessType,
            std::enable_if_t<(_Proc::EnableAlongStepGPIL), int> = 0>
  AlongStep(size_t _N, _Track *&_track, intmax_t *_doit_idx, _Tp *_doit_value,
            _Func *_doit_apply)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _value = mpl_type::GPIL(_track);
    if (_value < *_doit_value) {
      *_doit_idx   = _N;
      *_doit_value = _value;
      // Note: If one of the AlongStep 'wins' then we need to run all.
      // This is currently handled externally. (i.e. this apply function is ignored)
      *_doit_apply = [&]() { mpl_type::DoIt(_track); };
    }
  }

  //
  //  Invoked when a process does not provide a PIL proposal.
  //  Compiler will completely eliminate this "function call" in the binary
  //
  template <typename _Track, typename _Tp, typename _Func, typename _Proc = ProcessType,
            std::enable_if_t<!(_Proc::EnableAlongStepGPIL && _Proc::EnableAlongStepDoIt),
                             int> = 0>
  AlongStep(size_t, _Track *, intmax_t *, _Tp *, _Func *)
  {}

  //
  //  Invoked for selection among processes that DO NOT propose a PIL
  //  but DO have something to do (i.e. something that should always be done)
  //
  template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<(_Proc::EnableAlongStepGPIL && _Proc::EnableAlongStepDoIt),
                             int> = 0>
  explicit AlongStep(_Track *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    geantx::Log(kInfo) << GEANT_HERE << "[W/ GPIL ALONG-STEP DO-IT] "
                       << tim::demangle<_Proc>();

    if (IsAlive(*_track) && !IsStopped(*_track)) mpl_type::DoIt(_track);
    ///       if stopped
    ///          if alive && has-at-rest-processes
    ///             exec AtRest
    ///          return
  }

  //
  //  Invoked for anything that doesn't fit the conditions above.
  //  Compiler will completely eliminate this "function call" in the binary
  //
  template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<(!_Proc::EnableAlongStepGPIL && _Proc::EnableAlongStepDoIt),
                             int> = 0>
  explicit AlongStep(_Track *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    geantx::Log(kInfo) << GEANT_HERE << "[NO GPIL ALONG-STEP DO-IT] "
                       << tim::demangle<_Proc>();

    if (IsAlive(*_track) && !IsStopped(*_track)) mpl_type::DoIt(_track);
  }

  //
  //  Invoked when a process does not provide a PIL proposal nor a DoIt
  //  Compiler will completely eliminate this "function call" in the binary
  //
  template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<(!_Proc::EnableAlongStepGPIL && !_Proc::EnableAlongStepDoIt),
                             int> = 0>
  AlongStep(_Track *)
  {}
};

//======================================================================================//

template <typename ProcessType, typename ParticleType>
struct PostStep {
  using mpl_type  = mpl::PostStep<ProcessType, ParticleType>;
  using this_type = PostStep<ProcessType, ParticleType>;
  static constexpr bool EnableDoIt =
      ProcessType::EnableAlongStepDoIt && !ProcessType::EnablePostStepGPIL;

  template <typename _Track, typename _Tp, typename _Func, typename _Proc = ProcessType,
            std::enable_if_t<(_Proc::EnablePostStepGPIL), int> = 0>
  PostStep(size_t _N, _Track *_track, intmax_t *_doit_idx, _Tp *_doit_value,
           _Func *_doit_apply)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _value = mpl_type::GPIL(_track);
    if (_value < *_doit_value) {
      *_doit_idx   = _N;
      *_doit_value = _value;
      *_doit_apply = [=]() { mpl_type::DoIt(_track); };
    }
  }

  //
  //  Invoked when a process does not provide a PIL proposal.
  //  Compiler will completely eliminate this "function call" in the binary
  //
  template <typename _Track, typename _Tp, typename _Func, typename _Proc = ProcessType,
            std::enable_if_t<!(_Proc::EnablePostStepGPIL), int> = 0>
  PostStep(size_t, _Track *, intmax_t *, _Tp *, _Func *)
  {}

  //
  //  Invoked for selection among processes that DO NOT propose a PIL
  //  but DO have something to do (i.e. something that should always be done)
  //
  template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<(!_Proc::EnablePostStepGPIL && _Proc::EnablePostStepDoIt),
                             int> = 0>
  explicit PostStep(_Track *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    geantx::Log(kInfo) << GEANT_HERE << "[NO GPIL POST-STEP DO-IT] for "
                       << tim::demangle<_Proc>();
    mpl_type::DoIt(_track);
  }

  //
  //  Invoked for anything that doesn't fit the conditions above.
  //  Compiler will completely eliminate this "function call" in the binary
  //
  template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<!(!_Proc::EnablePostStepGPIL && _Proc::EnablePostStepDoIt),
                             int> = 0>
  explicit PostStep(_Track *)
  {}
};

//======================================================================================//

} // namespace geantx
