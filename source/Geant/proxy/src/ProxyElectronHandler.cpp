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
#include "Geant/proxy/ProxyElectronHandler.hpp"

namespace geantx {

GEANT_HOST
ProxyElectronHandler::ProxyElectronHandler() 
  : ProxyProcessHandler<ProxyElectronHandler>()
{
  fIonization     = new ProxyIonization();
  fBremsstrahlung = new ProxyBremsstrahlung();
}

GEANT_HOST_DEVICE
ProxyElectronHandler::~ProxyElectronHandler()
{
  delete fIonization;
  delete fBremsstrahlung;
}

GEANT_HOST_DEVICE
ProxyElectronHandler::ProxyElectronHandler(int tid)
  : ProxyProcessHandler<ProxyElectronHandler>(tid)
{
}

GEANT_HOST_DEVICE
void ProxyElectronHandler::GPIL(TrackState* track)
{
  ;
}

GEANT_HOST_DEVICE
void ProxyElectronHandler::DoIt(TrackState* track)
{
  ;
  //  track->fPhysicsProcessState[fProcessIndex].fNumOfInteractLengthLeft = -1;
  //  int nsec = this->fModel->SampleSecondaries(track);
}

}
