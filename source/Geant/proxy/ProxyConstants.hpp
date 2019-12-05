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

// Proxy2DVector (SeltzerBerger data) - valid up to maximumZ = 92
constexpr int maximumZ                = 92;
constexpr unsigned int numberOfXNodes = 32;
constexpr unsigned int numberOfYNodes = 57;

} // namespace geantx
