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
  static constexpr bool EnableAtRestGPIL    = true;
  static constexpr bool EnableAlongStepGPIL = true;
  static constexpr bool EnablePostStepGPIL  = true;
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
  ProxyTrackLimiter()  = default;
  ~ProxyTrackLimiter() = default;

  GEANT_HOST_DEVICE double AlongStepGPIL(const TrackState *)
  {
    return std::numeric_limits<double>::max();
  }
  GEANT_HOST_DEVICE double PostStepGPIL(const TrackState *)
  {
    return std::numeric_limits<double>::max();
  }
  GEANT_HOST_DEVICE double AtRestGPIL(const TrackState *)
  {
    return std::numeric_limits<double>::max();
  }

  GEANT_HOST_DEVICE void AlongStepDoIt(TrackState *) {}
  GEANT_HOST_DEVICE void PostStepDoIt(TrackState *) {}
  GEANT_HOST_DEVICE void AtRestDoIt(TrackState *) {}
};
} // namespace geantx