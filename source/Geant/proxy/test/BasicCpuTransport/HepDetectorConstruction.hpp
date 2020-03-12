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
 * @file Geant/proxy/test/BasicCpuTransport/HepDetectorConstruction.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/geometry/UserDetectorConstruction.hpp"

namespace geant {
inline namespace GEANT_IMPL_NAMESPACE {
  class UserDetectorConstruction;
  class RunManager;
} // namespace GEANT_IMPL_NAMESPACE
} // namespace geant

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {
  class Material;
}
} // namespace geantx

namespace userapplication {

class HepDetectorConstruction : public geantx::UserDetectorConstruction {
public:
  HepDetectorConstruction(geantx::RunManager *runmgr);

  virtual ~HepDetectorConstruction();

  // interface method to define the geometry for the application
  virtual void CreateGeometry();

  void SetGDMLFile(const std::string &gdml) { fGDMLFileName = gdml; }

  void DetectorInfo();

private:
  std::string fGDMLFileName;
};

} // namespace userapplication
