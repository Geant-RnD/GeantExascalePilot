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
class ProxyTrackLimiter : public Process {
public:
  // Enable/disable GetPhysicalInteractionLength (GPIL) functions
  static constexpr bool EnableAtRestGPIL    = false;
  static constexpr bool EnableAlongStepGPIL = false;
  static constexpr bool EnablePostStepGPIL  = false;
  // Enable/disable DoIt functions
  static constexpr bool EnableAtRestDoIt    = false;
  static constexpr bool EnableAlongStepDoIt = false;
  static constexpr bool EnablePostStepDoIt  = true;

  // for enable_if statements
  template <typename _Tp>
  static constexpr bool IsApplicable = std::is_base_of<Particle, _Tp>::value;

  // provide no specializations
  using specialized_types = Tuple<>;

public:
  using this_type = ProxyTrackLimiter;

  ProxyTrackLimiter()  = default;
  ~ProxyTrackLimiter() = default;

  GEANT_HOST_DEVICE void PostStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (_track->fHistoryState.fNsteps >= fMaxSteps) {
      geantx::Log(kInfo) << GEANT_HERE << "killing track: " << *_track;
      _track->fStatus = TrackStatus::Killed;
    }
  }

private:
  intmax_t fMaxSteps = 256 * get_rand();
};
} // namespace geantx