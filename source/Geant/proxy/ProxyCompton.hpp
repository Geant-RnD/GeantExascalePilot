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
#include "Geant/proxy/ProxyKleinNishina.hpp"

namespace geantx
{
class ProxyCompton : public ProxyEmProcess<ProxyCompton>
{
  friend class ProxyEmProcess<ProxyCompton>;
  ProxyKleinNishina *fModel = nullptr;    
public:
  // Enable/disable GetPhysicalInteractionLength (GPIL) functions
  static constexpr bool EnableAtRestGPIL    = false;
  static constexpr bool EnableAlongStepGPIL = false;
  static constexpr bool EnablePostStepGPIL  = true;
  // Enable/disable DoIt functions
  static constexpr bool EnableAtRestDoIt    = false;
  static constexpr bool EnableAlongStepDoIt = false;
  static constexpr bool EnablePostStepDoIt  = true;

  // for enable_if statements
  template <typename _Tp>
  static constexpr bool IsApplicable = std::is_base_of<Particle, _Tp>::value;
  
  // provide no specializations
  using specialized_types = std::tuple<>;
  
public:
  using this_type = ProxyCompton;
  
  ProxyCompton(){ fModel = new ProxyKleinNishina; }
  ~ProxyCompton() = default;
  
  double MacroscopicXSection(TrackState* _track)
  {  
    GEANT_THIS_TYPE_TESTING_MARKER("");
    //TODO: connectivity to Material
    int Z = 10;

    //TODO: implement MacroscopicCrossSection and CrossSectionPerAtom

    double xsection = fModel->CrossSectionPerAtom(Z, _track->fPhysicsState.fEkin);
    return xsection;
  }
  
  int FinalStateInteraction(int Z, double energy)
  {  
    GEANT_THIS_TYPE_TESTING_MARKER("");

    //update photon state and create an electron 
    int nsecondaries = fModel->SampleSecondaries(Z, energy);

    return nsecondaries;
  }

};
}  // namespace geantx
