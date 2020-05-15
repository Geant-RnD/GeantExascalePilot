//===------------------ GeantX --------------------------------------------===//
//                                                                              
// Geant Exascale Pilot                                                         
//                                                                              
// For the licensing terms see LICENSE file.                                    
// For the list of contributors see CREDITS file.                               
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.         
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyTypeDef.hpp                                          
 * @brief extended type definitions for proxy
 */
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __CUDA_ARCH__
  #define CONSTTYPE constexpr
#else
  #define CONSTTYPE __constant__
#endif

