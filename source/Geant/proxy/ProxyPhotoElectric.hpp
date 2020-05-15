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
 * @file Geant/proxy/ProxyPhotoElectric.hpp
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
#include "Geant/proxy/ProxySauterGavrila.hpp"

namespace geantx
{

class ProxyPhotoElectric;

template <> struct Model_traits<ProxyPhotoElectric>
{
  using Model_t = ProxySauterGavrila;
};

class ProxyPhotoElectric : public ProxyEmProcess<ProxyPhotoElectric>
{
  friend class ProxyEmProcess<ProxyPhotoElectric>;
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
  using this_type = ProxyPhotoElectric;
  
  GEANT_HOST
  ProxyPhotoElectric(){ this->fProcessIndex = kProxyPhotoElectric; }

  GEANT_HOST_DEVICE
  ProxyPhotoElectric(int tid) : ProxyEmProcess<ProxyPhotoElectric>(tid) 
  { this->fProcessIndex = kProxyPhotoElectric; }

  GEANT_HOST_DEVICE
  ~ProxyPhotoElectric() {};
  
  // mandatory methods for static polymorphism

  GEANT_HOST_DEVICE
  double GetLambda(int index, double energy) 
  {
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kLambdaPrim_phot_gamma)->Value(index,energy);
  } 

};



}  // namespace geantx
