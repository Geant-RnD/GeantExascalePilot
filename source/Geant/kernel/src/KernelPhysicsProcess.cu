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
 * @brief kernels for physics processes
 */
//===----------------------------------------------------------------------===//

//#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "Geant/track/TrackState.hpp"

#include "Geant/processes/Process.hpp"
#include "Geant/proxy/ProxyEmProcess.hpp"
#include "Geant/proxy/ProxyCompton.hpp"
#include "Geant/proxy/ProxyIonization.hpp"
#include "Geant/proxy/ProxyEmModel.hpp"
#include "Geant/proxy/ProxyKleinNishina.hpp"
#include "Geant/proxy/ProxyMollerScattering.hpp"
#include "Geant/proxy/ProxyPhysicsTableIndex.hpp"
#include "Geant/proxy/ProxyPhysicsVector.cuh"
#include "Geant/proxy/ProxyPhysicsTable.cuh"
#include "Geant/proxy/ProxyDataManager.cuh"

#ifndef ADDED_SUPPORT_FOR_SEPARABLE_COMPILATION
#include "Geant/processes/src/Process.cpp"
#include "Geant/proxy/src/ProxyEmProcess.cpp"
#include "Geant/proxy/src/ProxyCompton.cpp"
#include "Geant/proxy/src/ProxyEmModel.cpp"
#include "Geant/proxy/src/ProxyKleinNishina.cpp"
#include "Geant/proxy/src/ProxyPhysicsVector.cu"
#include "Geant/proxy/src/ProxyPhysicsTable.cu"
#endif

namespace geantx
{
// kernels

__global__
void gpil_gamma_process(int nstate, TrackPhysicsState* state_d, 
                        ProxyDataManager *dm_d) 
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  ProxyEmProcess<ProxyCompton> proc(tid);
  ProxyKleinNishina model(tid);

  proc.SetModel(&model);
  proc.SetDataManager(dm_d);

  while (tid < nstate) {
    // a temporary test
    double lambda = proc.GetLambda(1,state_d[tid].fEkin);
    // TODO: add a generic photon process and get GPIL for this step 
    // double step = proc.PostStepGPIL(state_d[tid]);

    tid += blockDim.x * gridDim.x;
  }
}

__global__
void gpil_electron_process(int nstate, TrackPhysicsState* state_d, 
                           ProxyDataManager *dm_d) 
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  ProxyEmProcess<ProxyIonization> proc(tid);
  ProxyMollerScattering model(tid);

  proc.SetModel(&model);
  proc.SetDataManager(dm_d);

  while (tid < nstate) {
    double lambda = proc.GetLambda(2,state_d[tid].fEkin);
    // TODO: add a generic electron process and get GPIL for this step 
    // double step = proc.PostStepGPIL(state_d[tid]);
    // ...
    tid += blockDim.x * gridDim.x;
  }
}

// wrappers

void GPILGammaProcess(int nstate, TrackPhysicsState* state_d, 
                      ProxyDataManager* dm_d, int nblocks, int nthreads, 
                      cudaStream_t stream)
{
  gpil_gamma_process<<<nblocks,nthreads,0,stream>>>(nstate, state_d, dm_d);
  cudaThreadSynchronize();
}


void GPILElectronProcess(int nstate, TrackPhysicsState* state_d, 
                         ProxyDataManager* dm_d, int nblocks, int nthreads, 
                         cudaStream_t stream)
{
  gpil_electron_process<<<nblocks,nthreads,0,stream>>>(nstate, state_d, dm_d);
  cudaThreadSynchronize();
}

}  // namespace geantx
