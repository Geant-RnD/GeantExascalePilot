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
class ProxyScattering : public Process {
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
  ProxyScattering()  = default;
  ~ProxyScattering() = default;

  // here the transportation proposed a step distance
  double AlongStepGPIL(const TrackState &state)
  {
    auto _safety = state.fGeometryState.fSafety;
    // inverse of the square of the safety
    return (1.0 / (_safety * _safety));
  }
  // double PostStepGPIL(const TrackState&);
  // double AtRestGPIL(const TrackState&);

  // here the transportation is applied
  void AlongStepDoIt(TrackState &state)
  {
    state.fDir.x() *= -0.5;
    state.fDir.y() *= -0.5;
    state.fDir.z() *= 0.25;
    state.fDir.Normalize();
  }
  void PostStepDoIt(TrackState &) {}
  // void AtRestDoIt(TrackState&);
};

} // namespace geantx