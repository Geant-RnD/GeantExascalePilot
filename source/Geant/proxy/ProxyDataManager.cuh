//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyDataManager.cuh
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyConstants.hpp"
#include "Geant/proxy/ProxyPhysicsTable.cuh"
#include "Geant/proxy/ProxyPhysicsTableIndex.hpp"

namespace geantx {

class ProxyDataManager {

  GEANT_HOST
  ProxyDataManager();

public:
  GEANT_HOST
  static ProxyDataManager *Instance();

  GEANT_HOST_DEVICE 
  ~ProxyDataManager();

  GEANT_HOST
  bool RetrievePhysicsData(/* const std::string& dataDirection */);

  GEANT_HOST
  bool RetrieveCutsTable(/* const std::string& dataDirection */);

  GEANT_HOST
  void RelocatePhysicsData(void *devPtr);

  GEANT_HOST
  inline int SizeOfObject() { return fSizeOfObject; }

  GEANT_HOST_DEVICE 
  ProxyPhysicsTable* GetTable(int index) { return fPhysicsTables[index]; }

  GEANT_HOST_DEVICE 
  double GetCutValue(int index, int ipart) { return fCutsTable[data::nParticleForCuts*ipart + index]; }

  GEANT_HOST_DEVICE 
  void Print();

  GEANT_HOST_DEVICE 
  void PrintCutsTable();

private:
  static ProxyDataManager *fInstance;

  int fSizeOfObject;

  // material cuts 
  int fNumOfCuts;      
  double *fCutsTable;

  int fNumPhysicsTables;
  ProxyPhysicsTable **fPhysicsTables;
};

} // namespace geantx

