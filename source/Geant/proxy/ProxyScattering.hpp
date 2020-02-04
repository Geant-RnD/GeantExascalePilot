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

#include "Geant/core/Logger.hpp"
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
  using specialized_types = std::tuple<CpuElectron>;

public:
  using this_type = ProxyScattering;

  ProxyScattering()  = default;
  ~ProxyScattering() = default;

  // here the transportation proposed a step distance
  double AlongStepGPIL(const TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");

    auto _safety = (5.0 * get_rand() - _track->fGeometryState.fSafety);
    if (_safety < 0.0) return std::numeric_limits<double>::max();

    // inverse of the square of the safety
    return (1.0 / (_safety * _safety));
  }

  // here the transportation is applied
  void AlongStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    ThreeVector rand = {get_rand(), get_rand(), get_rand()};
    rand.Normalize();
    _track->fDir = rand;
  }

  void PostStepDoIt(TrackState *_track) { GEANT_THIS_TYPE_TESTING_MARKER(""); }

  //------------------------------------------------------------------------------------//
  //
  //            Specialization for CpuElectron
  //
  //------------------------------------------------------------------------------------//
  template <typename _Tp,
            std::enable_if_t<(std::is_same<_Tp, CpuElectron>::value), int> = 0>
  double AlongStepGPIL(const TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return get_rand();
  }

  template <typename _Tp,
            std::enable_if_t<(std::is_same<_Tp, CpuElectron>::value), int> = 0>
  void AlongStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _dir        = _track->fDir;
    ThreeVector rand = {_dir.x() * get_rand(), _dir.y() * get_rand(), _dir.z()};
    rand.Normalize();
    _track->fDir = rand;
  }

  template <typename _Tp,
            std::enable_if_t<(std::is_same<_Tp, CpuElectron>::value), int> = 0>
  void PostStepDoIt(TrackState *_track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
  }
};

} // namespace geantx