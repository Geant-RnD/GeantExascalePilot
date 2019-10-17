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
#include "Geant/proxy/ProxyParticles.hpp"
#include "Geant/track/TrackState.hpp"
#include <map>

using namespace geantx;

template <typename T, typename U>
using map_t = std::map<T, U>;

using ParticleTypes = std::tuple<CpuGamma, CpuElectron, GpuGamma, GpuElectron>;

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessList {
  using particle = ParticleType;
  using physics  = std::tuple<ProcessTypes...>;
};

using CpuGammaPhysics = PhysicsProcessList<CpuGamma, ProxyScattering>;

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
