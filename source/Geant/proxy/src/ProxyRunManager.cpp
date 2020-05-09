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
 * @file Geant/proxy/src/ProxyRunManager.cpp
 * @brief the run manager for proxy physics validation 
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/ProxyRunManager.hpp"

namespace geantx {

template <typename T>
GEANT_HOST
ProxyRunManager<T>::ProxyRunManager() 
{
  fEventManager = new ProxyEventServer<T>();
  fDeviceManager = new ProxyDeviceManager<T>();
  fDeviceManager->SetEventManager(fEventManager);
}

template <typename T>
GEANT_HOST
ProxyRunManager<T>::~ProxyRunManager() 
{
  delete fEventManager;
}

} // namespace geantx
