//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyEventServer.cuh
 * @brief the event server
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyStackManager.hpp"

namespace geantx {

template <class TGenerator>
class ProxyEventServer {

public:

  GEANT_HOST
  ProxyEventServer();

  GEANT_HOST
  ~ProxyEventServer();

  //set an event generator
  GEANT_HOST
  void SetEventGenerator(TGenerator* generator) { fGenerator = generator; } 

  GEANT_HOST
  void GenerateEvents(int nevents);

  GEANT_HOST
  void ProcessEvents(int nevents);

  GEANT_HOST
  ProxyStackManager* GetStackManager() const { return fStackManager; }

  GEANT_HOST
  void PrintStackInfo();

private:
  TGenerator* fGenerator = nullptr;
  ProxyStackManager* fStackManager;
};

} // namespace geantx

