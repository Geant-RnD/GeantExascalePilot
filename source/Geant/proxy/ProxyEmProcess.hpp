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

  int fProcessIndex = kNullProcess;;
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
  
  ProxyEmProcess(){ 
    fRng = new ProxyRandom; 
    fModel = new  Model_t;
    fDataManager = ProxyDataManager::Instance(); 
  }

  ~ProxyEmProcess() = default;
  
  //mandatory methods
  double GetLambda(int index, double energy) { return static_cast<TEmProcess *>(this)->GetLambda(index,energy); }

  // the proposed along step physical interaction length
  double AlongStepGPIL(TrackState* _track);

  // the proposed post step physical interaction length
  double PostStepGPIL(TrackState* _track);

  // DoIt for the along step
  void AlongStepDoIt(TrackState* _track);

  // DoIt for the post step
  int PostStepDoIt(TrackState* _track);

  // Aux

};

template <typename TEmProcess>
double ProxyEmProcess<TEmProcess>::AlongStepGPIL(TrackState* track)
{
  GEANT_THIS_TYPE_TESTING_MARKER("");

  //the default stepLimit
  return std::numeric_limits<double>::max();
}

template <typename TEmProcess>
void ProxyEmProcess<TEmProcess>::AlongStepDoIt(TrackState* track)
{
  GEANT_THIS_TYPE_TESTING_MARKER("");
}

template <typename TEmProcess>
double ProxyEmProcess<TEmProcess>::PostStepGPIL(TrackState* track)
{
  GEANT_THIS_TYPE_TESTING_MARKER("");
  double step = std::numeric_limits<double>::max();

  double energy = track->fPhysicsState.fEkin;
  int index = track->fMaterialState.fMaterialId;

  double lambda = GetLambda(index, energy);

  //reset or update the number of the interaction length left  
  if ( track->fPhysicsProcessState.fNumOfInteractLengthLeft <= 0.0 ) {
    track->fPhysicsProcessState.fNumOfInteractLengthLeft = -vecCore::math::Log(fRng->uniform());
  }
  else {
    track->fPhysicsProcessState.fNumOfInteractLengthLeft
      -=  track->fPhysicsState.fPstep/track->fPhysicsProcessState.fPhysicsInteractLength;
  }    

  step = lambda * track->fPhysicsProcessState.fNumOfInteractLengthLeft;

  //save lambda and the current step
  track->fPhysicsProcessState.fPhysicsInteractLength = lambda;
  track->fPhysicsState.fPstep = step;

  return step;
}

template <typename TEmProcess>
int ProxyEmProcess<TEmProcess>::PostStepDoIt(TrackState* track)
{
  GEANT_THIS_TYPE_TESTING_MARKER("");

  int nsec = this->fModel->SampleSecondaries(track);

  return nsec;
}

}  // namespace geantx
