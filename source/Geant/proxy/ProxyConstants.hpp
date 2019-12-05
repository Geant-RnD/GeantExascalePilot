//===------------------ GeantX --------------------------------------------===//
//                                                                              
// Geant Exascale Pilot                                                         
//                                                                              
// For the licensing terms see LICENSE file.                                    
// For the list of contributors see CREDITS file.                               
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.         
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyConstants.hpp                                          
 * @brief constants for proxy physics processes and models
 */
//===----------------------------------------------------------------------===//

#pragma once

namespace geantx {

namespace brem {
  
// Proxy2DVector (SeltzerBerger data) - valid up to maximumZ = 92
constexpr int maximumZ                = 92;
constexpr unsigned int numberOfXNodes = 32;
constexpr unsigned int numberOfYNodes = 57;

constexpr double xgi[8] = {0.0199, 0.1017, 0.2372, 0.4083, 0.5917, 0.7628, 0.8983, 0.9801};
constexpr double wgi[8] = {0.0506, 0.1112, 0.1569, 0.1813, 0.1813, 0.1569, 0.1112, 0.0506};
} // namespace brem
  
} // namespace geantx
