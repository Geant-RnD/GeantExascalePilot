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
  ProxySecondaryGenerator()  = default;
  ~ProxySecondaryGenerator() = default;

  double AlongStepGPIL(const TrackState *) { return 0.0; }
  double PostStepGPIL(const TrackState *) { return 0.0; }
  double AtRestGPIL(const TrackState *) { return 0.0; }

  void AlongStepDoIt(TrackState *) {}
  void PostStepDoIt(TrackState *) {}
  void AtRestDoIt(TrackState *) {}
};
}