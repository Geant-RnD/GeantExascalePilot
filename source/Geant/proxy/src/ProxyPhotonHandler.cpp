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
#include "Geant/proxy/ProxyPhotonHandler.hpp"

namespace geantx {

GEANT_HOST
ProxyPhotonHandler::ProxyPhotonHandler() 
  : ProxyProcessHandler<ProxyPhotonHandler>()
{
  fCompton       = new ProxyCompton();
  fConversion    = new ProxyConversion();
  fPhotoElectron = new ProxyPhotoElectric();
}

GEANT_HOST_DEVICE
ProxyPhotonHandler::~ProxyPhotonHandler()
{
  delete fCompton;
  delete fConversion;
  delete fPhotoElectron;
}

GEANT_HOST_DEVICE
ProxyPhotonHandler::ProxyPhotonHandler(int tid)
  : ProxyProcessHandler<ProxyPhotonHandler>(tid)
{
}

GEANT_HOST_DEVICE
void ProxyPhotonHandler::GPIL(TrackState* track)
{
  ;
}

GEANT_HOST_DEVICE
void ProxyPhotonHandler::DoIt(TrackState* track)
{
  ;
  //  track->fPhysicsProcessState[fProcessIndex].fNumOfInteractLengthLeft = -1;
  //  int nsec = this->fModel->SampleSecondaries(track);
}

}
