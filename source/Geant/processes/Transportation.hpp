//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Transportation.hpp
 * @brief Transportation process
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/Profiler.hpp"
#include "Geant/core/Tuple.hpp"
#include "Geant/particles/Types.hpp"
#include "Geant/processes/Process.hpp"
#include "Geant/track/TrackAccessor.hpp"

#include <type_traits>

namespace geantx {
class Transportation : public Process {
public:
  // Enable/disable GetPhysicalInteractionLength (GPIL) functions
  static constexpr bool EnableAtRestGPIL    = false;
  static constexpr bool EnableAlongStepGPIL = true;
  static constexpr bool EnablePostStepGPIL  = false;
  // Enable/disable DoIt functions
  static constexpr bool EnableAtRestDoIt    = false;
  static constexpr bool EnableAlongStepDoIt = true;
  static constexpr bool EnablePostStepDoIt  = true;

  // for enable_if statements
  template <typename _Tp>
  static constexpr bool IsApplicable = std::is_base_of<Particle, _Tp>::value;

  // provide no specializations
  using specialized_types = Tuple<>;

public:
  using this_type = Transportation;

  Transportation();
  ~Transportation();

  // here the transportation proposed a step distance
  GEANT_HOST_DEVICE double AlongStepGPIL(const TrackState *)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return 10.0 * get_rand();
  }

  // here the transportation is applied
  GEANT_HOST_DEVICE void AlongStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    _track->fPos += _track->fPhysicsState.fPstep * _track->fDir;
  }
  GEANT_HOST_DEVICE void PostStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    // _track->fPos += _track->fPhysicsState.fPstep * _track->fDir;
  }
};

template <typename ParticleType>
struct ProcessAvailable<Transportation, ParticleType> : std::true_type {};

template <typename ParticleType>
struct ProcessEnabled<Transportation, ParticleType> : std::true_type {};

} // namespace geantx
