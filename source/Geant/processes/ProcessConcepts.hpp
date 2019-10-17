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

#include "Geant/core/Tuple.hpp"

#include <string>
#include <type_traits>

namespace geantx {
namespace mpl {
//--------------------------------------------------------------------------------------//
//
//                              AT-REST
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType>
struct AtRest {
  static_assert(std::is_base_of<Process, ProcessType>::value,
                "ProcessType must derive for Process");

  //----------------------------------------------------------------------------------//
  //                                  GPIL
  //----------------------------------------------------------------------------------//

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAtRestGPIL), int>           = 0,
            std::enable_if_t<!(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static double GPIL(ProcessType &p, _Args &&... args)
  {
    return p.AtRestGPIL(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAtRestGPIL), int>          = 0,
            std::enable_if_t<(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static double GPIL(ProcessType &p, _Args &&... args)
  {
    return p.AtRestGPIL<ParticleType>(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<!(ProcessType::EnableAtRestGPIL), int> = 0>
  static double GPIL(ProcessType &, _Args &&...)
  {}

  //----------------------------------------------------------------------------------//
  //                                  DO-IT
  //----------------------------------------------------------------------------------//

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAtRestDoIt), int>           = 0,
            std::enable_if_t<!(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static void DoIt(ProcessType &p, _Args &&... args)
  {
    p.AtRestDoIt(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAtRestDoIt), int>          = 0,
            std::enable_if_t<(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static void DoIt(ProcessType &p, _Args &&... args)
  {
    p.AtRestDoIt<ParticleType>(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<!(ProcessType::EnableAtRestDoIt), int> = 0>
  static void DoIt(ProcessType &, _Args &&...)
  {}
};

//--------------------------------------------------------------------------------------//
//
//                              Along-Step
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType>
struct AlongStep {
  static_assert(std::is_base_of<Process, ProcessType>::value,
                "ProcessType must derive for Process");

  //----------------------------------------------------------------------------------//
  //                                  GPIL
  //----------------------------------------------------------------------------------//

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAlongStepGPIL), int>        = 0,
            std::enable_if_t<!(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static double GPIL(ProcessType &p, _Args &&... args)
  {
    return p.AlongStepGPIL(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAlongStepGPIL), int>       = 0,
            std::enable_if_t<(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static double GPIL(ProcessType &p, _Args &&... args)
  {
    return p.AlongStepGPIL<ParticleType>(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<!(ProcessType::EnableAlongStepGPIL), int> = 0>
  static double GPIL(ProcessType &, _Args &&...)
  {}

  //----------------------------------------------------------------------------------//
  //                                  DO-IT
  //----------------------------------------------------------------------------------//

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAlongStepDoIt), int>        = 0,
            std::enable_if_t<!(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static void DoIt(ProcessType &p, _Args &&... args)
  {
    p.AlongStepDoIt(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnableAlongStepDoIt), int>       = 0,
            std::enable_if_t<(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static void DoIt(ProcessType &p, _Args &&... args)
  {
    p.AlongStepDoIt<ParticleType>(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<!(ProcessType::EnableAlongStepDoIt), int> = 0>
  static void DoIt(ProcessType &, _Args &&...)
  {}
};

//--------------------------------------------------------------------------------------//
//
//                              Post-Step
//
//--------------------------------------------------------------------------------------//

template <typename ProcessType>
struct PostStep {
  static_assert(std::is_base_of<Process, ProcessType>::value,
                "ProcessType must derive for Process");

  //----------------------------------------------------------------------------------//
  //                                  GPIL
  //----------------------------------------------------------------------------------//

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnablePostStepGPIL), int>         = 0,
            std::enable_if_t<!(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static double GPIL(ProcessType &p, _Args &&... args)
  {
    return p.PostStepGPIL(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnablePostStepGPIL), int>        = 0,
            std::enable_if_t<(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static double GPIL(ProcessType &p, _Args &&... args)
  {
    return p.PostStepGPIL<ParticleType>(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<!(ProcessType::EnablePostStepGPIL), int> = 0>
  static double GPIL(ProcessType &, _Args &&...)
  {}

  //----------------------------------------------------------------------------------//
  //                                  DO-IT
  //----------------------------------------------------------------------------------//

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnablePostStepDoIt), int>         = 0,
            std::enable_if_t<!(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static void DoIt(ProcessType &p, _Args &&... args)
  {
    p.PostStepDoIt(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<(ProcessType::EnablePostStepDoIt), int>        = 0,
            std::enable_if_t<(is_one_of_v<ParticleType, Specialized>), int> = 0>
  static void DoIt(ProcessType &p, _Args &&... args)
  {
    p.PostStepDoIt<ParticleType>(std::forward<_Args>(args)...);
  }

  template <typename ParticleType, typename... _Args,
            typename Specialized = ProcessType::specialized_types,
            std::enable_if_t<!(ProcessType::EnablePostStepDoIt), int> = 0>
  static void DoIt(ProcessType &, _Args &&...)
  {}
};

//--------------------------------------------------------------------------------------//
} // namespace mpl

} // namespace geantx
