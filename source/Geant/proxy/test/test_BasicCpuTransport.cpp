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

#include "Geant/geometry/RunManager.hpp"
#include "Geant/geometry/UserDetectorConstruction.hpp"
#include "Geant/core/Config.hpp"
#include "Geant/core/Memory.hpp"
#include "Geant/processes/Transportation.hpp"
#include "Geant/proxy/ProxyScattering.hpp"
#include "Geant/proxy/ProxySecondaryGenerator.hpp"
#include "Geant/proxy/ProxyStepLimiter.hpp"
#include "Geant/proxy/ProxyTrackLimiter.hpp"
#include "Geant/track/TrackState.hpp"

using namespace geantx;

void initialize_geometry() {}

void initialize_physics() {}

TrackState *get_primary_particle()
{
  return nullptr;
}

int main(int argc, char **argv)
{
  initialize_geometry();
  initialize_physics();
}
