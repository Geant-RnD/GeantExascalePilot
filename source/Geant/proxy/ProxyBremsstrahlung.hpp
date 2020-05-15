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
  static constexpr bool EnableAlongStepDoIt = false;
  static constexpr bool EnablePostStepDoIt  = true;

  // for enable_if statements
  template <typename _Tp>
  static constexpr bool IsApplicable = std::is_base_of<Particle, _Tp>::value;
  
  // provide no specializations
  using specialized_types = std::tuple<>;
  
public:
  using this_type = ProxyBremsstrahlung;
  
  GEANT_HOST
  ProxyBremsstrahlung() { this->fProcessIndex = kProxyBremsstrahlung; }

  GEANT_HOST_DEVICE
  ProxyBremsstrahlung(int tid) : ProxyEmProcess<ProxyBremsstrahlung>(tid)  
  { this->fProcessIndex = kProxyBremsstrahlung; }

  GEANT_HOST_DEVICE
  ~ProxyBremsstrahlung() {}

  // mandatory methods for static polymorphism
  GEANT_HOST_DEVICE
  double GetLambda(int index, double energy) 
  {
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kLambda_eBrem_eminus)->Value(index,energy);
  } 

  // auxiliary 
  GEANT_HOST_DEVICE
  double GetDEDX(int index, double energy) 
  { 
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kDEDX_eBrem_eminus)->Value(index,energy);
  }

};

}  // namespace geantx
