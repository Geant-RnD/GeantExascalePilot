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
 * @file Geant/proxy/test/BasicCpuTransport/HepDetectorConstruction.cpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//
#include "HepDetectorConstruction.hpp"

#include "Geant/geometry/RunManager.hpp"
//#include "VecGeom/gdml/Frontend.h"

#include "Geant/geometry/RunManager.hpp"

#include "Geant/track/Types.hpp"


namespace userapplication {

HepDetectorConstruction::HepDetectorConstruction(geantx::RunManager *runmgr)
    : geantx::UserDetectorConstruction(runmgr), fGDMLFileName("cms2018.gdml")
{
}

HepDetectorConstruction::~HepDetectorConstruction() {}

void HepDetectorConstruction::CreateGeometry()
{
  std::cout << "  **** LOADING GEOMETRY FROM GDML = " << fGDMLFileName << std::endl;

  /*
  auto const loaded = vgdml::Frontend::Load(fGDMLFileName);

  if (!loaded) std::cout << "*** Fail to Load GDML ***" << std::endl;
  else DetectorInfo();
  */
}

void HepDetectorConstruction::DetectorInfo() 
{
  /*
  std::cout << "\n ====    Detector Info    ===== " << std::endl;
  std::cout << "Have depth " << vecgeom::GeoManager::Instance().getMaxDepth() << std::endl;

  auto &geoManager = vecgeom::GeoManager::Instance();
  std::vector<vecgeom::LogicalVolume *> v1;
  geoManager.GetAllLogicalVolumes(v1);
  std::cout << "Have logical volumes " << v1.size() << std::endl;

  std::vector<vecgeom::VPlacedVolume *> v2;
  geoManager.getAllPlacedVolumes(v2);
  std::cout << "Have placed volumes " << v2.size() << std::endl;
  */
}

} // namespace userapplication
