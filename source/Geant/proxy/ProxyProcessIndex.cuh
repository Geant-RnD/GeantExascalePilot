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
 * @file Geant/proxy/ProxyProcessIndex.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//
#pragma once

#include <string>

namespace geantx {

enum ProxyProcessIndex {
  kNullProcess = -1,          // 
  kProxyBremsstrahlung,       // e- Bremsstrahlung
  kProxyIonization,           // e- Ionization
  kProxyMSC,                  // e- Multiple Scattering
  kProxyCompton,              // gamma Compton
  kProxyConversion,           // gamma Conversion (pair production)
  kProxyPhotoElectric,        // gamma PhotoElectricEffect
  kNumberProxyProcess         // number of physics processes    
};

const std::string ProxyProcessName[kNumberProxyProcess] = {
  "eBrem",
  "eIoni",
  "eMsc",
  "gComp",
  "gConv",
  "gPhot"
};

} // namespace geantx
