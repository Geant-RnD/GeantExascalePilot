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
 * @file Geant/proxy/src/ProxyDeviceManager.cpp
 * @brief a device manager for proxy physics validation
 */
//===----------------------------------------------------------------------===//
//          

#include "Geant/proxy/ProxyDeviceManager.hpp"
#include "Geant/kernel/KernelPhysicsProcess.cuh"

namespace geantx {

template <typename T>
GEANT_HOST
ProxyDeviceManager<T>::ProxyDeviceManager(int nb, int nt) 
  : fNBlocks(nb), fNThreads(nt), fPerformance(false)
{
  Initialize();
}

template <typename T>
GEANT_HOST
ProxyDeviceManager<T>::~ProxyDeviceManager()
{
  cudaStreamDestroy(fStream);
  cudaEventDestroy(fStart);
  cudaEventDestroy(fStop);
}

template <typename T>
GEANT_HOST
void ProxyDeviceManager<T>::Initialize()
{
  int nDevice;
  cudaGetDeviceCount(&nDevice);
  
  if(nDevice > 0) {
    cudaDeviceReset();
    std::cout << "... CUDA Kernel<<<" << fNBlocks << "," 
	      << fNThreads <<">>> (...) ..." << std::endl;
  }
  else {
    std::cout << "No Cuda Capable Device ... " << std::endl;
  }    

  //initialize the stream
  cudaStreamCreate(&fStream);

  //create cuda events
  cudaEventCreate(&fStart);
  cudaEventCreate(&fStop);
}

template <typename T>
GEANT_HOST
void ProxyDeviceManager<T>::SetBlockThread(int nblocks, int nthreads)
{
  fNBlocks  = nblocks;
  fNThreads = nthreads;
} 

template <typename T>
GEANT_HOST
float ProxyDeviceManager<T>::StopTimer()
{
  float elapsedTime = 0.0;
  cudaEventRecord (fStop,0);
  cudaEventSynchronize (fStop);
  cudaEventElapsedTime (&elapsedTime,fStart,fStop);
  return elapsedTime;
} 

template <typename T>
GEANT_HOST
void ProxyDeviceManager<T>::DownloadPhysicsState(int nstate, TrackPhysicsState* state_o, TrackPhysicsState* state_d)
{
  int msize = nstate*sizeof(TrackPhysicsState);
  cudaMalloc((void**)&state_o, msize);
  cudaMemcpyAsync(state_o, state_d, msize, cudaMemcpyDeviceToHost, fStream);
}

template <typename T>
GEANT_HOST
void ProxyDeviceManager<T>::UploadTrackData()
{
  //debugging
  fEventManager->PrintStackInfo();

  if(fPerformance) StartTimer();

  //photons
  TrackStack* tracks = fEventManager->GetStackManager()->GetPhotonStack();

  fNPhotons = tracks->size();
  if(fNPhotons > 0) {
    int msize = fNPhotons*sizeof(TrackPhysicsState);

    //allocate and copy to GPU
    cudaHostAlloc((void**) &fPhotons_h, msize, cudaHostAllocDefault);

    for(int i = 0 ; i < fNPhotons ; ++i){
      fPhotons_h[i] = (tracks->Get(i))->fPhysicsState;
    }

    cudaMalloc((void**)&fPhotons_d, msize);
    cudaMemcpyAsync(fPhotons_d, fPhotons_h, msize, cudaMemcpyHostToDevice, fStream);
  }

  //electrons
  tracks = fEventManager->GetStackManager()->GetElectronStack();
  fNElectrons = tracks->size();

  if(fNElectrons > 0) {
    int msize = fNElectrons*sizeof(TrackPhysicsState);

    //allocate and copy to GPU
    cudaHostAlloc((void**) &fElectrons_h, msize, cudaHostAllocDefault);

    for(int i = 0 ; i < fNElectrons ; ++i){
      fElectrons_h[i] = (tracks->Get(i))->fPhysicsState;
    }

    cudaMalloc((void**)&fElectrons_d, msize);
    cudaMemcpyAsync(fElectrons_d, fElectrons_h, msize, cudaMemcpyHostToDevice, fStream);
  }

  //TODO: pre-allocated memory for secondaries 

  if(fPerformance) {
    float elapsedTimeH2D = StopTimer();
    printf("UploadTrackData: elapsedTimeD2H = %f ms\n", elapsedTimeH2D);
  }
}

template <typename T>
GEANT_HOST
void ProxyDeviceManager<T>::DeallocateTrackData() {
  cudaFree(fPhotons_d);
  cudaFree(fElectrons_d);

  cudaFree(fPhotons_h);
  cudaFree(fElectrons_h);

  //validation
  cudaFree(fPhotons_o);
  cudaFree(fElectrons_o);
}

template <typename T>
GEANT_HOST
void ProxyDeviceManager<T>::DownloadTrackData()
{
  if(fPerformance) StartTimer();

  DownloadPhysicsState(fNPhotons, fPhotons_o, fPhotons_d);
  DownloadPhysicsState(fNElectrons, fElectrons_o, fElectrons_d);

  if(fPerformance) { 
    float elapsedTimeD2H = StopTimer();
    printf("DownloadTrackData: elapsedTimeD2H = %f ms\n", elapsedTimeD2H);
  }
}

template <typename T>
void ProxyDeviceManager<T>::DoStep()
{
  if(fPerformance) StartTimer();

  ProxyDataManager * dm = ProxyDataManager::Instance();
  ProxyDataManager *dm_d;
  cudaMalloc((void **)&dm_d, dm->SizeOfObject());
  dm->RelocatePhysicsData(dm_d);

  //prepare random engines on the device
  // cudaMalloc(&fRandomStates, fNBlocks*fNThreads* sizeof(curandState));
  // curand_setup_gpu(fRandomStates, time(NULL), fNBlocks, fNThreads);

  GPILElectronProcess(fNElectrons, fElectrons_d, dm_d, fNBlocks, fNThreads, fStream);

  GPILGammaProcess(fNPhotons, fPhotons_d, dm_d, fNBlocks, fNThreads, fStream);

  //  cudaFree(fRandomStates);
 
  if(fPerformance) {
    float elapsedTimeGPU = StopTimer();
    printf("DoStep: elapsedTimeGPU = %f ms\n", elapsedTimeGPU);
  }
}

} // namespace geantx 
