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
class ProxyStepLimiter : public Process {
public:
  // Enable/disable GetPhysicalInteractionLength (GPIL) functions
  static constexpr bool EnableAtRestGPIL    = false;
  static constexpr bool EnableAlongStepGPIL = false;
  static constexpr bool EnablePostStepGPIL  = false;
  // Enable/disable DoIt functions
  static constexpr bool EnableAtRestDoIt    = false;
  static constexpr bool EnableAlongStepDoIt = true;
  static constexpr bool EnablePostStepDoIt  = false;

  // for enable_if statements
  template <typename _Tp>
  static constexpr bool IsApplicable = std::is_base_of<Particle, _Tp>::value;

  // provide no specializations
  using specialized_types = Tuple<>;

public:
  using this_type = ProxyStepLimiter;

  ProxyStepLimiter()  = default;
  ~ProxyStepLimiter() = default;

  GEANT_HOST_DEVICE void AlongStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (_track->fPhysicsState.fPstep > fStepLimit()) {
      geantx::Log(kInfo) << GEANT_HERE
                         << "limiting step: " << _track->fPhysicsState.fPstep << " -> "
                         << fStepLimit();
      _track->fPhysicsState.fPstep = fStepLimit();
    }
  }

private:
  static double fStepLimit()
  {
    static double _value = tim::get_env<double>("GEANT_STEP_LIMIT_FACTOR", 5.0);
    return _value;
  }
};
} // namespace geantx