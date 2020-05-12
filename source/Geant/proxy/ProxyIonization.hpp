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

#pragma once

#include "Geant/core/Logger.hpp"
#include "Geant/core/Tuple.hpp"
#include "Geant/particles/Types.hpp"
#include "Geant/processes/Process.hpp"

#include "Geant/proxy/ProxyEmProcess.hpp"
#include "Geant/proxy/ProxyMollerScattering.hpp"
#include "Geant/proxy/ProxyConstants.hpp"

namespace geantx
{

class ProxyIonization;

template <> struct Model_traits<ProxyIonization>
{
  using Model_t = ProxyMollerScattering;
};

class ProxyIonization : public ProxyEmProcess<ProxyIonization>
{
  friend class ProxyEmProcess<ProxyIonization>;
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
  using this_type = ProxyIonization;
  
  ProxyIonization() { this->fProcessIndex = kProxyIonization; }

  ~ProxyIonization() = default;

  // mandatory methods for static polymorphism

  GEANT_HOST_DEVICE
  double GetLambda(int index, double energy) 
  {
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kLambda_eIoni_eminus)->Value(index,energy);
  } 

  // specialization for the ionization process
  GEANT_HOST_DEVICE
  double AlongStepGPIL(TrackState* _track);

  GEANT_HOST_DEVICE
  void AlongStepDoIt(TrackState* _track);
  
  // auxiliary 
  GEANT_HOST_DEVICE
  double GetDEDX(int index, double energy) 
  { 
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kDEDX_eIoni_eminus)->Value(index,energy);
  }

  GEANT_HOST_DEVICE
  double GetRange(int index, double energy) 
  { 
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kRange_eIoni_eminus)->Value(index,energy);
  }

  GEANT_HOST_DEVICE
  double GetInverseRange(int index, double energy) 
  { 
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kInverseRange_eIoni_eminus)->Value(index,energy);
  }

};

GEANT_HOST_DEVICE
double ProxyIonization::AlongStepGPIL(TrackState* track)
{
  double stepLimit = DBL_MAX;

  int index = track->fMaterialState.fMaterialId;
  double energy = track->fPhysicsState.fEkin;

  double range = GetRange(index,energy);
  double minE = this->fModel->GetLowEnergyLimit();
  if(energy < minE) range *= energy/minE;

  //production cuts index: electron = 1
  double cut = fDataManager->GetCutValue(index,1);

  double finR = (data::finalRange < cut) ? data::finalRange : cut;
  stepLimit = (range > finR) ? range*data::dRoverRange + finR*(1.0-data::dRoverRange)*(2.0*finR/range) : range;

  return stepLimit;

}

GEANT_HOST_DEVICE
void ProxyIonization::AlongStepDoIt(TrackState* track)
{
  //  GEANT_THIS_TYPE_TESTING_MARKER("");

  double stepLength = track->fPhysicsState.fPstep;
  double energy = track->fPhysicsState.fEkin;
  int index = track->fMaterialState.fMaterialId;

  double range = GetRange(index,energy);
  double minE = this->fModel->GetLowEnergyLimit();

  if (stepLength >= range || energy <= minE) {
    track->fPhysicsState.fEkin = 0.0;
    track->fPhysicsState.fEdep += energy;
    return;
  }

  double dedx = GetDEDX(index,energy);
  if(energy<minE) dedx *= vecCore::math::Sqrt(energy/minE);
  double eloss = stepLength * dedx;

  if(eloss > energy*data::linLossLimit) {
    range -= stepLength;
    eloss = energy - GetInverseRange(index,range);
  }

  //energy  balance
  double finalE = energy - eloss;
  if(finalE < minE) {
    eloss += finalE;
    finalE = 0.0;    
  }

  eloss = vecCore::math::Max(eloss, 0.0);
  track->fPhysicsState.fEkin = finalE;
  track->fPhysicsState.fEdep += eloss;

}

}  // namespace geantx
