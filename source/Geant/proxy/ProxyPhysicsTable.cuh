//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyPhysicsTable.cuh
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyPhysicsVector.cuh"

namespace geantx {

class ProxyPhysicsTable {

 public:
  GEANT_HOST_DEVICE 
  ProxyPhysicsTable();

  GEANT_HOST_DEVICE 
  ~ProxyPhysicsTable();

  GEANT_HOST
  void Relocate(void *devPtr);

  GEANT_HOST_DEVICE 
  void Print();

  GEANT_HOST
  bool RetrievePhysicsTable(const std::string& fileName);

private:
  int fNumPhysicsVector;
  ProxyPhysicsVector **fPhysicsVectors;
};

} // namespace geantx

