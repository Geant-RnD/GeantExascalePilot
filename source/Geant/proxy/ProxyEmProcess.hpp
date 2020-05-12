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
 * @file Geant/proxy/ProxyEmProcess.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/Logger.hpp"
#include "Geant/core/Tuple.hpp"
#include "Geant/particles/Types.hpp"
#include "Geant/processes/Process.hpp"

#include "Geant/proxy/ProxyRandom.hpp"
#include "Geant/proxy/ProxyDataManager.cuh"
#include "Geant/proxy/ProxyProcessIndex.cuh"
#include "Geant/proxy/ProxyPhysicsTableIndex.hpp"

namespace geantx
{

template <typename TEmProcess> struct Model_traits;

template <typename TEmProcess>
class ProxyEmProcess : public Process
{
protected:

  using Model_t = typename Model_traits<TEmProcess>::Model_t;

  int fProcessIndex = -1;
  Model_t *fModel = nullptr;
  ProxyRandom *fRng = nullptr;
  ProxyDataManager *fDataManager = nullptr;

public:
  // Enable/disable GetPhysicalInteractionLength (GPIL) functions

  static constexpr bool EnableAtRestGPIL    = false;
  static constexpr bool EnableAlongStepGPIL = true;
  static constexpr bool EnablePostStepGPIL  = true;
  // Enable/disable DoIt functions
  static constexpr bool EnableAtRestDoIt    = false;
  static constexpr bool EnableAlongStepDoIt = true;
  static constexpr bool EnablePostStepDoIt  = true;
  
  // for enable_if statements
  template <typename _Tp>
  static constexpr bool IsApplicable = std::is_base_of<Particle, _Tp>::value;
  
  // provide no specializations
  using specialized_types = std::tuple<>;
  
public:
  using this_type = ProxyEmProcess;
  
  GEANT_HOST
  ProxyEmProcess()
  { 
    fRng = new ProxyRandom; 
    fModel = new  Model_t;
    fDataManager = ProxyDataManager::Instance(); 
  }

  GEANT_HOST_DEVICE
  ProxyEmProcess(int tid) : Process(tid) {}

  GEANT_HOST_DEVICE
  ~ProxyEmProcess() {}

  GEANT_HOST_DEVICE
  void SetModel(Model_t *model) { fModel = model; }

  GEANT_HOST_DEVICE
  void SetDataManager(ProxyDataManager *manager) { fDataManager = manager; }
  
  //mandatory methods
  GEANT_HOST_DEVICE
  double GetLambda(int index, double energy) 
  { return static_cast<TEmProcess *>(this)->GetLambda(index,energy); }

  // the proposed along step physical interaction length
  GEANT_HOST_DEVICE
  double AlongStepGPIL(TrackState* _track)
  { return static_cast<TEmProcess *>(this)->AlongStepGPIL(_track); }

  // the proposed post step physical interaction length
  GEANT_HOST_DEVICE
  double PostStepGPIL(TrackState* _track);

  // DoIt for the along step
  GEANT_HOST_DEVICE
  void AlongStepDoIt(TrackState* _track);

  // DoIt for the post step
  GEANT_HOST_DEVICE
  int PostStepDoIt(TrackState* _track);

  // Aux

};

template <typename TEmProcess>
GEANT_HOST_DEVICE
void ProxyEmProcess<TEmProcess>::AlongStepDoIt(TrackState* track)
{
  //  GEANT_THIS_TYPE_TESTING_MARKER("");
  ;
}

template <typename TEmProcess>
GEANT_HOST_DEVICE
double ProxyEmProcess<TEmProcess>::PostStepGPIL(TrackState* track)
{
  //  GEANT_THIS_TYPE_TESTING_MARKER("");
  //  double step = std::numeric_limits<double>::max();
  double step = DBL_MAX;

  double energy = track->fPhysicsState.fEkin;
  int index = track->fMaterialState.fMaterialId;

  index =2;
  double lambda = GetLambda(index, energy);

  //reset or update the number of the interaction length left  
  if ( track->fPhysicsProcessState[fProcessIndex].fNumOfInteractLengthLeft <= 0.0 ) {
    track->fPhysicsProcessState[fProcessIndex].fNumOfInteractLengthLeft = -vecCore::math::Log(fRng->uniform());
  }
  else {
    track->fPhysicsProcessState[fProcessIndex].fNumOfInteractLengthLeft
      -=  track->fPhysicsState.fPstep/track->fPhysicsProcessState[fProcessIndex].fPhysicsInteractLength;
  }    

  step = track->fPhysicsProcessState[fProcessIndex].fNumOfInteractLengthLeft/lambda;

  //save lambda and the current step
  track->fPhysicsProcessState[fProcessIndex].fPhysicsInteractLength = 1.0/lambda;

  return step;
}

template <typename TEmProcess>
GEANT_HOST_DEVICE
int ProxyEmProcess<TEmProcess>::PostStepDoIt(TrackState* track)
{
  //  GEANT_THIS_TYPE_TESTING_MARKER("");

  track->fPhysicsProcessState[fProcessIndex].fNumOfInteractLengthLeft = -1;
  int nsec = this->fModel->SampleSecondaries(track);

  return nsec;
}

}  // namespace geantx
