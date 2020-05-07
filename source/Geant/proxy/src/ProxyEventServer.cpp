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
 * @file Geant/proxy/ProxyEventServer.cpp
 * @brief the event server for proxy physics validation
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/ProxyEventServer.hpp"
#include "Geant/proxy/ProxyEvent.hpp"

namespace geantx {

template <typename T>
GEANT_HOST
ProxyEventServer<T>::ProxyEventServer() 
{
  fStackManager = new ProxyStackManager();
}

template <typename T>
GEANT_HOST
ProxyEventServer<T>::~ProxyEventServer() 
{
  delete fStackManager;
}

template <typename T>
GEANT_HOST
void ProxyEventServer<T>::GenerateEvents(int nevents) 
{
  for(int i = 0 ; i < nevents ; ++i) {
    ProxyEvent* evt = fGenerator->GenerateOneEvent();
    fStackManager->AddEvent(evt);    
  }
  //push an event to the Event Stack
}

template <typename T>
GEANT_HOST
void ProxyEventServer<T>::ProcessEvents(int nevents) 
{
  //TODO: add a workflow to process events
  ;
}

template <typename T>
GEANT_HOST
void ProxyEventServer<T>::PrintStackInfo() 
{
  std::cout << "ProxyEventServer::PrintStackInfo "   << std::endl;
  std::cout << "N photons   = " << fStackManager->GetNumberOfPhoton()   << std::endl;
  std::cout << "N electrons = " << fStackManager->GetNumberOfElectron() << std::endl;
  std::cout << "N others    = " << fStackManager->GetNumberOfOther()    << std::endl;
}

} // namespace geantx
