
#include <iostream>
#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>

#include "Geant/proxy/ProxyIsotope.cuh"
#include "Geant/proxy/ProxyIsotopeTable.cuh"
#include "Geant/proxy/ProxyElement.cuh"
#include "Geant/proxy/ProxyElementTable.cuh"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"

#include "Geant/proxy/ProxyProcess.cuh"
//#include "Geant/proxy/ProxyEmProcess.cuh"
//#include "Geant/proxy/ProxyCompton.cuh"

using namespace geantx;

__global__ void kernel_process()
{
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  printf("GPU tid = %d\n", tid);

  ProxyProcess proc(tid);
  proc.Print();
}

int main() {

  // set the default number of threads and thread blocks
  int theNBlocks  =  2;
  int theNThreads =  4;

  ProxyProcess* proc = new ProxyProcess();
  proc->Print();

  kernel_process<<<theNBlocks,theNThreads>>>();
  cudaDeviceSynchronize();

  delete proc;

  return 0;
}
