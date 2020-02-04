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
#include "navigation/NavigationState.h"
#include "navigation/GlobalLocator.h"
#include "base/Vector3D.h"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

namespace NavigationInterface {

/** @brief Function for navigation that checks if location is the same or not
 *
 *  @param track TrackState to be checked and updated with the new NavigationPath
 *  @param tmpstate Temporary navigation state to be used internally (to avoid redundant
 * allocations)
 *  @return true if the location is still the same
 */
VECCORE_ATT_HOST_DEVICE
bool IsSameLocation(TrackState &track, VolumePath_t &tmpstate)
{
  //#### NOT USING YET THE NEW NAVIGATORS ####//
  using Vector3D_t = vecgeom::Vector3D<vecgeom::Precision>;

  // TODO: not using the direction yet here !!
  bool samepath = vecgeom::GlobalLocator::HasSamePath(
      Vector3D_t(track.fPos.x(), track.fPos.y(), track.fPos.z()),
      *track.fGeometryState.fPath, tmpstate);

  if (!samepath) {
    tmpstate.CopyTo(track.fGeometryState.fNextpath);
#ifdef VECGEOM_CACHED_TRANS
    track.fGeometryState.fNextPath->UpdateTopMatrix();
#endif
  }
  return samepath;
}

} // namespace NavigationInterface
} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx