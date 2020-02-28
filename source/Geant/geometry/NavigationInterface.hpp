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
 * @brief Interface between a Track object and the VecGeom navigator.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/Typedefs.hpp"
#include "Geant/track/TrackState.hpp"

// VecGeom
#include "VecCore/VecMath.h"
#include "VecGeom/navigation/VNavigator.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/navigation/GlobalLocator.h"
#include "VecGeom/base/Vector3D.h"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

namespace NavigationInterface {

  /** @brief Function for navigation that checks if location is the same or not 
   *  
   *  @param track TrackState to be checked and updated with the new NavigationPath
   *  @param tmpstate Temporary navigation state to be used internally (to avoid redundant allocations)
   *  @return true if the location is still the same
  */
  VECCORE_ATT_HOST_DEVICE
  bool IsSameLocation(TrackState &track, VolumePath_t &tmpstate)
  {
    //#### NOT USING YET THE NEW NAVIGATORS ####//
    using Vector3D_t = vecgeom::Vector3D<vecgeom::Precision>;

    // TODO: not using the direction yet here !!
    bool samepath = vecgeom::GlobalLocator::HasSamePath(Vector3D_t(track.fPos.x(), track.fPos.y(), track.fPos.z()), *track.fGeometryState.fPath, tmpstate);

    if (!samepath) {
      tmpstate.CopyTo(track.fGeometryState.fNextpath);
#ifdef VECGEOM_CACHED_TRANS
      track.fGeometryState.fNextPath->UpdateTopMatrix();
#endif
    }
    return samepath;
  }

  //______________________________________________________________________________
  VECCORE_ATT_HOST_DEVICE
  void FindNextBoundary(TrackState &track)
  {
    constexpr double gTolerance = 1.e-9;
    // back-up the pre-step point boundary flag
    bool onboundary = track.fGeometryState.fIsOnBoundaryPreStp = track.fGeometryState.fBoundary;

    // Find distance to next boundary, within proposed step.
    typedef vecgeom::Vector3D<vecgeom::Precision> Vector3D_t;

    // Retrieve navigator for the track
    vecgeom::VNavigator const *newnav = track.fGeometryState.fVolume->GetNavigator();
    // Check if current safety allows for the proposed step
    double safety = track.fGeometryState.fSafety;
    const double pstep = track.fPhysicsState.fPstep;
    if (safety > pstep) {
      track.fGeometryState.fSnext = pstep;
      track.fGeometryState.fBoundary = (false);
      return;
    }
    const double snext  = newnav->ComputeStepAndSafety(
        Vector3D_t(track.fPos.x(), track.fPos.y(), track.fPos.z()), Vector3D_t(track.fDir.x(), track.fDir.y(), track.fDir.z()),
        vecCore::math::Min<double>(1.E20, pstep), *track.fGeometryState.fPath, !onboundary, safety);
    track.fGeometryState.fBoundary = (snext < pstep);
    track.fGeometryState.fSnext = (vecCore::math::Max<double>(2. * gTolerance, snext + 2. * gTolerance));
    track.fGeometryState.fSafety = (vecCore::math::Max<double>(safety, 0));
  }


} // NavigationInterface
} // GEANT_IMPL_NAMESPACE
} // geantx
