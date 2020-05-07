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
 * @file Geant/proxy/src/ProxyStackManager.cpp
 * @brief a track stack manager for proxy physics validation
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/ProxyStackManager.hpp"

namespace geantx {

GEANT_HOST
ProxyStackManager::ProxyStackManager() 
{
  fPhotonStack = new TrackStack;
  fElectronStack = new TrackStack;
  fOtherStack = new TrackStack;
}

GEANT_HOST
ProxyStackManager::~ProxyStackManager() 
{
  fPhotonStack->clear();
  fElectronStack->clear();
  fOtherStack->clear();
}

GEANT_HOST
void ProxyStackManager::AddEvent(ProxyEvent* evt) 
{
  int nvertex = evt->GetNumberOfVertices();  
  for(int i = 0 ; i < nvertex ; ++i) {
    ProxyVertex* vtx = evt->GetVertexVector()->Get(i);
    int nparticle = vtx->GetNumberOfParticles();
    
    for(int j = 0 ; j < nparticle ; ++j) {
      TrackState* part = vtx->GetParticleVector()->Get(j);
      // photon
      if(part->fSchedulingState.fGVcode == 22) {
        fPhotonStack->push_back(part);
      }
      // electron
      else if (part->fSchedulingState.fGVcode == 11) {
        fElectronStack->push_back(part);
      }
      else {
        fOtherStack->push_back(part);
      }
    }
  }

}

} // namespace geantx
