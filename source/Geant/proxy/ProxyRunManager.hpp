//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyRunManager.cuh
 * @brief the run manager for proxy physics validation
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyEventServer.hpp"
#include "Geant/proxy/ProxyEventGenerator.hpp"
#include "Geant/proxy/ProxyDeviceManager.hpp"

namespace geantx {

template <typename TGenerator> struct Generator_traits;

template <class TGenerator>
class ProxyRunManager 
{
  GEANT_HOST
  ProxyRunManager();

  ProxyRunManager(const ProxyRunManager& obj) = delete;
  ProxyRunManager& operator=(const ProxyRunManager& obj) = delete;

  using Generator_t = typename Generator_traits<TGenerator>::Generator_t;
  using EventManager_t = ProxyEventServer<Generator_t>;
  using DeviceManager_t = ProxyDeviceManager<Generator_t>;

public:
  GEANT_HOST
  static ProxyRunManager* Instance();

  GEANT_HOST
  ~ProxyRunManager();

  //number of events
  void SetNumberOfEvents(size_t nevents) { fNumberOfEvents = nevents; }

  size_t GetNumberOfEvents() const { return fNumberOfEvents; }

  //set an event generator to EventManager
  void RegisterEventGenerator(Generator_t* gen) 
    { fEventManager->SetEventGenerator(gen); }

  //accessor to the event manager
  EventManager_t* GetEventManager() const { return fEventManager; }

  //accessor to the device manager
  DeviceManager_t* GetDeviceManager() const { return fDeviceManager; }

  //TODO: add detector/geometry construction 

private:
  static ProxyRunManager *fInstance;

  size_t fNumberOfEvents = 0;
  EventManager_t* fEventManager = nullptr;
  DeviceManager_t* fDeviceManager = nullptr;

};

} // namespace geantx

