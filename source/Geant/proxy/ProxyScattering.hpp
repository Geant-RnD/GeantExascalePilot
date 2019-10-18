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
#include "Geant/core/Logger.hpp"

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
  using specialized_types = std::tuple<CpuElectron>;

public:
  ProxyScattering()  = default;
  ~ProxyScattering() = default;

  // here the transportation proposed a step distance
  double AlongStepGPIL(const TrackState *_track)
  {
    auto _safety = _track->fGeometryState.fSafety;
    geantx::Log(kInfo) << GEANT_HERE << "calc GPIL: " << *_track;

    // inverse of the square of the safety
    return (1.0 / (_safety * _safety));
  }
  // double PostStepGPIL(const TrackState*);
  // double AtRestGPIL(const TrackState*);

  // here the transportation is applied
  void AlongStepDoIt(TrackState *_track)
  {
    _track->fDir.x() += 0.5;
    _track->fDir.y() -= 0.1;
    _track->fDir.z() += 0.2;
    _track->fDir.Normalize();
    geantx::Log(kInfo) << GEANT_HERE << "apply along: " << *_track;
  }
  void PostStepDoIt(TrackState *_track)
  {
    geantx::Log(kInfo) << GEANT_HERE << "apply post: " << *_track;
  }
  // void AtRestDoIt(TrackState*);

  //------------------------------------------------------------------------------------//
  //
  //            Specialization for CpuElectron
  //
  //------------------------------------------------------------------------------------//
  template <typename _Tp,
            std::enable_if_t<(std::is_same<_Tp, CpuElectron>::value), int> = 0>
  double AlongStepGPIL(const TrackState *_track)
  {
    geantx::Log(kInfo) << GEANT_HERE << "[CPU_ELECTRON] calc GPIL: " << *_track;
    return std::numeric_limits<double>::max();
  }

  template <typename _Tp,
            std::enable_if_t<(std::is_same<_Tp, CpuElectron>::value), int> = 0>
  void AlongStepDoIt(TrackState *_track)
  {
    _track->fPos += 1.0;
    geantx::Log(kInfo) << GEANT_HERE << "[CPU_ELECTRON] apply along: " << *_track;
  }

  template <typename _Tp,
            std::enable_if_t<(std::is_same<_Tp, CpuElectron>::value), int> = 0>
  void PostStepDoIt(TrackState *_track)
  {
    geantx::Log(kInfo) << GEANT_HERE << "[CPU_ELECTRON] apply post: " << *_track;
  }
};

} // namespace geantx