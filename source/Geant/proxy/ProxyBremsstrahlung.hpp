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
 * @file Geant/proxy/ProxyBremsstrahlung.hpp
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
#include "Geant/proxy/ProxySeltzerBerger.hpp"

namespace geantx
{

class ProxyBremsstrahlung;

template <> struct Model_traits<ProxyBremsstrahlung>
{
  using Model_t = ProxySeltzerBerger;
};

class ProxyBremsstrahlung : public ProxyEmProcess<ProxyBremsstrahlung>
{
  friend class ProxyEmProcess<ProxyBremsstrahlung>;
public:
  // Enable/disable GetPhysicalInteractionLength (GPIL) functions
  static constexpr bool EnableAtRestGPIL    = false;
  static constexpr bool EnableAlongStepGPIL = false;
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
  using this_type = ProxyBremsstrahlung;
  
  ProxyBremsstrahlung(){ }
  ~ProxyBremsstrahlung() = default;

  // the proposed along step physical interaction length
  double AlongStepGPIL(TrackState* _track);

  int FinalStateInteraction(TrackState* _track)
  {  
    GEANT_THIS_TYPE_TESTING_MARKER("");

    //update electron state and create an electron 
    int nsecondaries = this->fModel->SampleSecondaries(_track);

    return nsecondaries;
  }
};

double ProxyBremsstrahlung::AlongStepGPIL(TrackState* track)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    double stepLimit = std::numeric_limits<double>::max();

    int index = track->fMaterialState.fMaterialId;
    double energy = track->fPhysicsState.fEkin;

    double range = fDataManager->GetTable(ProxyPhysicsTableIndex::kRange_eIoni_eminus)->Value(index,energy);
    double minE = this->fModel->GetLowEnergyLimit();
    if(energy < minE) range *= energy/minE;

    //production cuts index: electron = 1
    double cut = fDataManager->GetCutValue(index,1);

    double finR = vecCore::math::Min(data::finalRange, cut);

    stepLimit = (range > finR) ? range*data::dRoverRange + finR*(1.0-data::dRoverRange)*(2.0*finR/range) : range;

    return stepLimit;
  }


}  // namespace geantx
