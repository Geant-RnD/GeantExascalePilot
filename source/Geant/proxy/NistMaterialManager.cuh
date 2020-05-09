//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/NistMaterialManager.hpp
 * @brief the NIST material manager
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxySingleton.hpp"
#include "Geant/proxy/ProxyMaterial.cuh"

#include <string>

#include <iostream>
#include <cassert>

namespace geantx {

struct NistElementPrimitive {
  int fZ;
  double fWeight;
};

struct NistMaterial {
  std::string fName;
  double fDensity;
  int    fZ;
  double fMeanExcitationEnergy;
  int fNumberOfElements;
  int fState;
  NistElementPrimitive *fElement;
};

class NistMaterialManager : public ProxySingleton<NistMaterialManager>
{
  GEANT_HOST
  NistMaterialManager();

  friend class ProxySingleton<NistMaterialManager>;

public:

  GEANT_HOST
  ~NistMaterialManager() {}

  // NIST Elemenets

  GEANT_HOST
  inline int GetNumberOfMaterials() const { return fNumberOfMaterials; }  

  GEANT_HOST
  void Print();

  GEANT_HOST
  void PrintMaterialList();

private:

  GEANT_HOST
  bool LoadData();

private:

  int fNumberOfMaterials;
  NistMaterial* fMaterial;
};

} // namespace geantx
