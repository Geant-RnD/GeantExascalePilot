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

enum EletonProcessIndex {
  kNullElectronProcess = -1,  // 
  kProxyBremsstrahlung,       // e- Bremsstrahlung
  kProxyIonization,           // e- Ionization
  kProxyMSC,                  // e- Multiple Scattering
  kNumberElectronProcess         // number of physics processes    
};

enum GammaProcessIndex {
  kNullGammaProcess = -1,     // 
  kProxyCompton,              // gamma Compton
  kProxyConversion,           // gamma Conversion (pair production)
  kProxyPhotoElectric,        // gamma PhotoElectricEffect
  kNumberGammaProcess         // number of physics processes    
};

const std::string ElectronProcessName[kNumberElectronProcess] = {
  "eBrem",
  "eIoni",
  "eMsc"
};

const std::string GammaProcessName[kNumberGammaProcess] = {
  "gComp",
  "gConv",
  "gPhot"
};

} // namespace geantx
