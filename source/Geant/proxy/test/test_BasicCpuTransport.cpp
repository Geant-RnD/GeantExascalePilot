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

#include "volumes/PlacedVolume.h"
#include "volumes/LogicalVolume.h"
#include "management/GeoManager.h"
#include "volumes/Box.h"
#include "volumes/Orb.h"
#include "volumes/Trapezoid.h"

using namespace geantx;
using namespace vecgeom;
//::VPlacedVolume;
//using vecgeom::LogicalVolume;

template <typename T, typename U>
using map_t = std::map<T, U>;

using ParticleTypes = std::tuple<CpuGamma, CpuElectron, GpuGamma, GpuElectron>;

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessList {
  using particle = ParticleType;
  using physics  = std::tuple<ProcessTypes...>;
};

using CpuGammaPhysics = PhysicsProcessList<CpuGamma, ProxyScattering>;
//VPlacedVolume *
void initialize_geometry()
{
  UnplacedBox *worldUnplaced      = new UnplacedBox(10, 10, 10);
  UnplacedTrapezoid *trapUnplaced = new UnplacedTrapezoid(4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0);
  UnplacedBox *boxUnplaced        = new UnplacedBox(2.5, 2.5, 2.5);
  UnplacedOrb *orbUnplaced        = new UnplacedOrb(2.8);

  LogicalVolume *world = new LogicalVolume("world", worldUnplaced);
  LogicalVolume *trap  = new LogicalVolume("trap", trapUnplaced);
  LogicalVolume *box   = new LogicalVolume("box", boxUnplaced);
  LogicalVolume *orb   = new LogicalVolume("orb", orbUnplaced);

  Transformation3D *ident = new Transformation3D(0, 0, 0, 0, 0, 0);
  orb->PlaceDaughter("orb1", box, ident);
  trap->PlaceDaughter("box1", orb, ident);

  Transformation3D *placement1 = new Transformation3D(5, 5, 5, 0, 0, 0);
  Transformation3D *placement2 = new Transformation3D(-5, 5, 5, 0, 0, 0);   // 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D(5, -5, 5, 0, 0, 0);   // 0, 45,  0);
  Transformation3D *placement4 = new Transformation3D(5, 5, -5, 0, 0, 0);   // 0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5, 5, 0, 0, 0);  // 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-5, 5, -5, 0, 0, 0);  // 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D(5, -5, -5, 0, 0, 0);  // 0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-5, -5, -5, 0, 0, 0); // 45, 45, 45);

  world->PlaceDaughter("trap1", trap, placement1);
  world->PlaceDaughter("trap2", trap, placement2);
  world->PlaceDaughter("trap3", trap, placement3);
  world->PlaceDaughter("trap4", trap, placement4);
  world->PlaceDaughter("trap5", trap, placement5);
  world->PlaceDaughter("trap6", trap, placement6);
  world->PlaceDaughter("trap7", trap, placement7);
  world->PlaceDaughter("trap8", trap, placement8);

  VPlacedVolume *w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();
  //return w;
}

void initialize_physics() {}
TrackState *get_primary_particle()
{
  return nullptr;
}

int main(int argc, char **argv)
{
  initialize_geometry();

  // basic geometry checks
  LogicalVolume const* logWorld = GeoManager::Instance().GetWorld()->GetLogicalVolume();
  if (logWorld) {
    // print detector information
    logWorld->PrintContent();
    std::cout <<"\n # placed volumes: "<< logWorld->GetNTotal() << "\n";
  }

  initialize_physics();
}
