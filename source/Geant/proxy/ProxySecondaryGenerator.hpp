//
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
//
/**
 * @file
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/Tuple.hpp"
#include "Geant/particles/Types.hpp"
#include "Geant/processes/Process.hpp"

namespace geantx {
class ProxySecondaryGenerator : public Process {
public:
  // Enable/disable GetPhysicalInteractionLength (GPIL) functions
  static constexpr bool EnableAtRestGPIL    = false;
  static constexpr bool EnableAlongStepGPIL = false;
  static constexpr bool EnablePostStepGPIL  = false;
  // Enable/disable DoIt functions
  static constexpr bool EnableAtRestDoIt    = true;
  static constexpr bool EnableAlongStepDoIt = true;
  static constexpr bool EnablePostStepDoIt  = true;

  // for enable_if statements
  template <typename _Tp>
  static constexpr bool IsApplicable = std::is_base_of<Particle, _Tp>::value;

  // provide no specializations
  using specialized_types = Tuple<>;

public:
  using this_type = ProxySecondaryGenerator;

  ProxySecondaryGenerator()  = default;
  ~ProxySecondaryGenerator() = default;

  GEANT_HOST_DEVICE void AtRestDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
  }

  GEANT_HOST_DEVICE void AlongStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
  }

  GEANT_HOST_DEVICE void PostStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    geantx::Log(kInfo) << GEANT_HERE << "post step do it secondary: " << *_track;
    if (_track->fStatus == TrackStatus::Alive) {
      if (10.0 * get_rand() < get_rand()) {
        geantx::Log(kInfo) << GEANT_HERE << "killing track: " << *_track;
        _track->fStatus = TrackStatus::Killed;
      }
    }
  }
};
} // namespace geantx