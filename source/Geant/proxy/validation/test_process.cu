
#include <iostream>
#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>

#include "Geant/proxy/ProxyPhysicalConstants.hpp"

#include "Geant/processes/Process.hpp"
#include "Geant/proxy/ProxyEmProcess.hpp"
#include "Geant/proxy/ProxyCompton.hpp"
#include "Geant/proxy/ProxyEmModel.hpp"
#include "Geant/proxy/ProxyKleinNishina.hpp"

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
#include "Geant/proxy/src/ProxyDataManager.cu"

#endif

using namespace geantx;

__host__ __device__ void validation_process(double lambda) 
{
  //TODO: add validation items with input TrackState and check GPIL
  printf("Lambda = %g\n", lambda);
}

void cpu_process()
{
  // a process on the host
  ProxyEmProcess<ProxyCompton> *proc = new ProxyEmProcess<ProxyCompton>();

  //TODO: add validation items with input TrackState and check GPIL
  validation_process(proc->GetLambda(1,10));

  delete proc;
}

__global__ void gpu_process(ProxyDataManager *dm_d)
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  ProxyEmProcess<ProxyCompton> proc(tid);
  ProxyKleinNishina model(tid);

  proc.SetModel(&model);
  proc.SetDataManager(dm_d);

  //TODO: add validation items with input TrackState and check GPIL
  validation_process(proc.GetLambda(1,10));
}

int main(int argc, char* argv[]) {

  // set the default number of threads and thread blocks
  int theNBlocks  =  1;
  int theNThreads =  1;

  if(argc >= 2) theNBlocks = atoi(argv[1]);
  if(argc >= 3) theNThreads = atoi(argv[2]);

  ProxyDataManager * dm = ProxyDataManager::Instance();

  // relocate ProxyDataManager to the device
  // TODO: put this step into an initialization of a coprocessor manager
  ProxyDataManager *dm_d;
  cudaMalloc((void **)&dm_d, dm->SizeOfObject());
  dm->RelocatePhysicsData(dm_d);

  // CPU kernel
  cpu_process();

  // GPU kernel
  gpu_process<<<theNBlocks,theNThreads>>>(dm_d);
  cudaDeviceSynchronize();

  cudaFree(dm_d);

  return 0;
}
