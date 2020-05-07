//===------------------ GeantX --------------------------------------------===//
//                                                                              
// Geant Exascale Pilot                                                         
//                                                                              
// For the licensing terms see LICENSE file.                                    
// For the list of contributors see CREDITS file.                               
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.         
//===----------------------------------------------------------------------===//
/**                                                                             
 * @file Geant/proxy/KernelPhysicsProcess.cuh                                    
 * @brief kernels of physics processes
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackState.hpp"
#include "Geant/proxy/ProxyDataManager.cuh"

namespace geantx {

void GPILGammaProcess(int nstate, TrackPhysicsState* state_d, 
                      ProxyDataManager* dm_d, int nblocks, int nthreads, 
                      cudaStream_t stream);

void GPILElectronProcess(int nstate, TrackPhysicsState* state_d, 
                         ProxyDataManager* dm_d, int nblocks, int nthreads, 
                         cudaStream_t stream);

} // namespace geantx                                                           
