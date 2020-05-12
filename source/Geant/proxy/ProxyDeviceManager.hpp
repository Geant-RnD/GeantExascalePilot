//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyDeviceManager.cuh
 * @brief a device manager for proxy physics validation
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackState.hpp"
#include "Geant/proxy/ProxyEventServer.hpp"
#include "Geant/proxy/ProxyVector.cuh"
#include "Geant/proxy/ProxyDataManager.cuh"

namespace geantx {

template <class T>
class ProxyDeviceManager 
{
  using TrackStack = ProxyVector<TrackState*>;
  using EventManager_t = ProxyEventServer<T>;
public:

  GEANT_HOST
  ProxyDeviceManager(int NBlocks = 1, int NThreads = 1);

  GEANT_HOST
  ~ProxyDeviceManager();

  GEANT_HOST
  void Initialize();

  GEANT_HOST
  void SetBlockThread(int nblocks, int nthreads);

  GEANT_HOST
  void SetEventManager(EventManager_t* mgr) { fEventManager = mgr; }

  GEANT_HOST
  void UploadTrackData();

  GEANT_HOST
  void DeallocateTrackData();

  GEANT_HOST
  void UploadTrackState();

  GEANT_HOST
  void DeallocateTrackState();

  GEANT_HOST
  void DownloadTrackData();

  GEANT_HOST
  void DoStep();

  GEANT_HOST
  void DoGPIL();

  //for performance measurements
  void SetPerformanceFlag(bool flag) { fPerformance = flag; };

private:

  //for validation
  GEANT_HOST
  void DownloadPhysicsState(int nstate, TrackPhysicsState* state_o, 
                            TrackPhysicsState* state_d);

  void StartTimer() { cudaEventRecord (fStart,0); }
  float StopTimer();

private:

  int fNBlocks;
  int fNThreads;

  cudaStream_t fStream;

  TrackPhysicsState* fPhotons_d;
  TrackPhysicsState* fElectrons_d;

  TrackPhysicsState* fPhotons_h;
  TrackPhysicsState* fElectrons_h;

  TrackState** fPhotonState_d;
  TrackState** fElectronState_d;

  EventManager_t* fEventManager = nullptr;
  ProxyDataManager* fDataManager_d = nullptr;

  //for validation
  int fNPhotons = 0;
  int fNElectrons = 0;
  TrackPhysicsState* fPhotons_o;
  TrackPhysicsState* fElectrons_o;

  //for performance
  bool fPerformance = false;
  cudaEvent_t fStart;
  cudaEvent_t fStop;
};

} // namespace geantx 
