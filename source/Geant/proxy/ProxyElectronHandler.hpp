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
 * @file Geant/proxy/ProxyElectronHandler.hpp
 * @brief ProxyProcessHandler for electron processes
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/proxy/ProxyProcessHandler.hpp"

#include "Geant/proxy/ProxyIonization.hpp"
#include "Geant/proxy/ProxyBremsstrahlung.hpp"

namespace geantx
{

class ProxyElectronHandler : ProxyProcessHandler<ProxyElectronHandler>
{
  ProxyIonization* fIonization = nullptr;
  ProxyBremsstrahlung* fBremsstrahlung = nullptr;

  friend class ProxyProcessHandler<ProxyElectronHandler>;

public:

  GEANT_HOST
  ProxyElectronHandler();

  GEANT_HOST_DEVICE
  ProxyElectronHandler(int tid);

  GEANT_HOST_DEVICE
  ~ProxyElectronHandler();

  // the proposed step physical interaction length and 
  GEANT_HOST_DEVICE
  void GPIL(TrackState* _track);

  // DoIt for the along step
  GEANT_HOST_DEVICE
  void DoIt(TrackState* _track);
};

}  // namespace geantx
