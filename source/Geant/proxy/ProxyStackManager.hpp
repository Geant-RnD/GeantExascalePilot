//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyStackManager.hpp
 * @brief a stack manager of primary tracks
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackState.hpp"
#include "Geant/proxy/ProxyEvent.hpp"
#include "Geant/proxy/ProxyVector.cuh"

namespace geantx {

class ProxyStackManager {

  using TrackStack = ProxyVector<TrackState*>;

public:

  ProxyStackManager();

  ~ProxyStackManager();

  int GetNumberOfPhoton() const { return fPhotonStack->size(); }

  int GetNumberOfElectron() const { return fElectronStack->size(); }

  int GetNumberOfOther() const { return fOtherStack->size(); }

  TrackStack* GetPhotonStack() const { return fPhotonStack; }

  TrackStack* GetElectronStack() const { return fElectronStack; }

  TrackStack* GetOtherStack() const { return fOtherStack; }

  void AddEvent(ProxyEvent* event);

private:
  TrackStack* fPhotonStack;
  TrackStack* fElectronStack;
  TrackStack* fOtherStack;
};

} // namespace geantx
