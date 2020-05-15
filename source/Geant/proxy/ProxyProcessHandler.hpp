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
 * @file Geant/proxy/ProxyProcessHandler.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/track/TrackState.hpp"
#include "Geant/proxy/ProxyRandom.hpp"
#include "Geant/proxy/ProxyDataManager.cuh"
#include "Geant/proxy/ProxyProcessIndex.cuh"

namespace geantx
{

template <typename Handler>
class ProxyProcessHandler
{

protected:
  int fProcessIndex = -1;
  ProxyRandom *fRng = nullptr;
  ProxyDataManager *fDataManager = nullptr;

public:

  GEANT_HOST
  ProxyProcessHandler();

  GEANT_HOST_DEVICE
  ProxyProcessHandler(int tid);

  GEANT_HOST_DEVICE
  ~ProxyProcessHandler();

  GEANT_HOST_DEVICE
  void SetDataManager(ProxyDataManager *manager) { fDataManager = manager; }
  
  // the proposed step physical interaction length
  GEANT_HOST_DEVICE
  void GPIL(TrackState* _track) 
  { static_cast<Handler *>(this)->GPIL(_track); }

  // DoIt for the selected process
  GEANT_HOST_DEVICE
  void DoIt(TrackState* _track)
  { static_cast<Handler *>(this)->DoIt(_track); }

};

template <typename Handler>
GEANT_HOST
ProxyProcessHandler<Handler>::ProxyProcessHandler()
{ 
  fRng = new ProxyRandom; 
  fDataManager = ProxyDataManager::Instance(); 
}

template <typename Handler>
GEANT_HOST_DEVICE
ProxyProcessHandler<Handler>::~ProxyProcessHandler()
{ 
  delete fRng;
  delete fDataManager;
}

template <typename Handler>
GEANT_HOST_DEVICE
ProxyProcessHandler<Handler>::ProxyProcessHandler(int tid)
{ 
  fProcessIndex = -1;
  //get device data manager
  ;
}

}  // namespace geantx
