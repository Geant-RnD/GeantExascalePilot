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
 * @file Geant/proxy/ProxyPhotonHandler.hpp
 * @brief ProxyProcessHandler for photon processes
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/proxy/ProxyProcessHandler.hpp"

#include "Geant/proxy/ProxyCompton.hpp"
#include "Geant/proxy/ProxyConversion.hpp"
#include "Geant/proxy/ProxyPhotoElectric.hpp"

namespace geantx
{

class ProxyPhotonHandler : ProxyProcessHandler<ProxyPhotonHandler>
{
  ProxyCompton* fCompton = nullptr;
  ProxyConversion* fConversion = nullptr;
  ProxyPhotoElectric* fPhotoElectron = nullptr;

  friend class ProxyProcessHandler<ProxyPhotonHandler>;

public:

  GEANT_HOST
  ProxyPhotonHandler();

  GEANT_HOST_DEVICE
  ProxyPhotonHandler(int tid);

  GEANT_HOST_DEVICE
  ~ProxyPhotonHandler();

  // the proposed step physical interaction length and 
  GEANT_HOST_DEVICE
  void GPIL(TrackState* _track);

  // DoIt for the along step
  GEANT_HOST_DEVICE
  void DoIt(TrackState* _track);
};

}  // namespace geantx
