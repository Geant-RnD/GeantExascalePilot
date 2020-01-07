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
 * @file Geant/proxy/ProxyConversion.hpp
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
#include "Geant/proxy/ProxyBetheHeitler.hpp"

namespace geantx
{

class ProxyConversion;

template <> struct Model_traits<ProxyConversion>
{
  using Model_t = ProxyBetheHeitler;
};

class ProxyConversion : public ProxyEmProcess<ProxyConversion>
{
  friend class ProxyEmProcess<ProxyConversion>;
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
  using this_type = ProxyConversion;
  
  ProxyConversion(){ this->fProcessIndex = kProxyConversion; }
  ~ProxyConversion() = default;
  
  // mandatory methods for static polymorphism

  double GetLambda(int index, double energy) 
  {
    return fDataManager->GetTable(ProxyPhysicsTableIndex::kLambda_conv_gamma)->Value(index,energy);
  } 

};



}  // namespace geantx
