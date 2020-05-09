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
 * @file Geant/proxy/src/ProxyElementTable.cu
 * @brief the element table
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/ProxyElementTable.cuh"

namespace geantx {

GEANT_HOST
void ProxyElementTable::Relocate(void* devPtr)
{
  //capacity() fData

  printf("relocating ElementTable size = %d capacity=%d\n", size(), capacity());

  //element vector
  ProxyElement** h_fElementVector = fData;
  
  int nsize = size();

  // device pointers in host memory
  ProxyElement *vector_d[nsize];

  size_t bufferSize = 0;
  size_t vectorSize = 0;

  for (int i = 0 ; i < nsize ; ++i) {
     vectorSize += Get(i)->MemorySize();
     cudaMalloc((void **)&(vector_d[i]), Get(i)->MemorySize());    
     Get(i)->Relocate(vector_d[i]);
  }

  ProxyElement** d_fElementVector;
  cudaMalloc((void **)&d_fElementVector, vectorSize);

  // copy the pointer to alias table pointers from the host to the device
  cudaMemcpy(d_fElementVector, vector_d, vectorSize, cudaMemcpyHostToDevice);

  fData = d_fElementVector;

  // copy the manager from host to device.
  bufferSize = vectorSize + 2*sizeof(size_t) + sizeof(bool);
  cudaMemcpy(devPtr, this, bufferSize, cudaMemcpyHostToDevice);

  // persistency
  fData = h_fElementVector;


  
}

} // namespace geantx
