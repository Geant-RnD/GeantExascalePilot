//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Interface between the scheduler or the field integrator.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackAccessor.hpp"
#include "Geant/track/TrackState.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

class TaskData;

class FieldLookup {
public:
  using ThreeVector_t = vecgeom::Vector3D<double>;

  VECCORE_ATT_HOST_DEVICE
  static void GetFieldValue(const ThreeVector_t &pos, ThreeVector_t &magFld, double &bmag)
  {
    constexpr double kTeslaToKiloGauss = 10.0;
    magFld = ThreeVector_t{kTeslaToKiloGauss, kTeslaToKiloGauss, kTeslaToKiloGauss};
    bmag   = 3;
  }
};

class FieldPropagationHandler {
public:
  using ThreeVector_t = vecgeom::Vector3D<double>;

  FieldPropagationHandler()  = default;
  ~FieldPropagationHandler() = default;

  VECCORE_ATT_HOST_DEVICE
  double Curvature(const TrackState &track) const;

  VECCORE_ATT_HOST_DEVICE
  double Curvature(const TrackState &track, const ThreeVector_t &magFld,
                   double bmag) const;

  VECCORE_ATT_HOST_DEVICE
  void DoIt(TrackState &track, TaskData *td) const;

  VECCORE_ATT_HOST_DEVICE
  void PropagateInVolume(TrackState &track, double crtstep,
                         const ThreeVector &BfieldInitial, double bmag,
                         TaskData *td) const;

  VECCORE_ATT_HOST_DEVICE
  bool IsSameLocation(TrackState &track, TaskData *td) const;

  VECCORE_ATT_HOST_DEVICE
  GEANT_FORCE_INLINE
  double SafeLength(const TrackState &track, double eps, const ThreeVector_t &magFld,
                    double bmag) const
  {
    // Returns the propagation length in field such that the propagated point is
    // shifted less than eps with respect to the linear propagation.
    // OLD: return 2. * sqrt(eps / track.Curvature(Bz));
    double c   = Curvature(track, magFld, bmag); //, td);
    double val = 0.0;
    // if (c < 1.E-10) { val= 1.E50; } else
    val = 2. * sqrt(eps / c);
    return val;
  }
};

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx