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
 * @file Geant/proxy/src/ProxyMaterialTable.cu
 * @brief the material table 
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/ProxyMaterialTable.cuh"

namespace geantx {

GEANT_HOST
void ProxyMaterialTable::Relocate(void* devPtr)
{
  //capacity() fData

  printf("relocating MaterialTable size = %d capacity=%d\n", size(), capacity());

  //element vector
  ProxyMaterial** h_fMaterialVector = fData;
  
  int nsize = size();

  // device pointers in host memory
  ProxyMaterial *vector_d[nsize];

  size_t bufferSize = 0;
  size_t vectorSize = 0;

  for (int i = 0 ; i < nsize ; ++i) {
     vectorSize += Get(i)->MemorySize();
     cudaMalloc((void **)&(vector_d[i]), Get(i)->MemorySize());    
     Get(i)->Relocate(vector_d[i]);
  }

  ProxyMaterial** d_fMaterialVector;
  cudaMalloc((void **)&d_fMaterialVector, vectorSize);

  // copy the pointer to alias table pointers from the host to the device
  cudaMemcpy(d_fMaterialVector, vector_d, vectorSize, cudaMemcpyHostToDevice);

  fData = d_fMaterialVector;

  // copy the manager from host to device.
  bufferSize = vectorSize + 2*sizeof(size_t) + sizeof(bool);
  cudaMemcpy(devPtr, this, bufferSize, cudaMemcpyHostToDevice);

  // persistency
  fData = h_fMaterialVector;

}

} // namespace geantx
