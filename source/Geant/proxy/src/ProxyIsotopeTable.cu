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
 * @file Geant/proxy/src/ProxyIsotopeTable.cu
 * @brief the isotope table
 */
//===----------------------------------------------------------------------===//
//

#include "Geant/proxy/ProxyIsotopeTable.cuh"

namespace geantx {

GEANT_HOST
void ProxyIsotopeTable::Relocate(void* devPtr)
{
  //isotope vector on host
  ProxyIsotope** h_fIsotopeVector = fData;
  
  int nsize = size();
  ProxyIsotope** d_fIsotopeVector;
  cudaMalloc((void **)&d_fIsotopeVector, nsize*sizeof(ProxyIsotope));

  // device pointers in host memory
  ProxyIsotope *vector_d[nsize];

  size_t bufferSize = 0;
  size_t vectorSize = 0;

  for (int i = 0 ; i < nsize ; ++i) {
     vectorSize += Get(i)->MemorySize();
     cudaMalloc((void **)&(vector_d[i]), sizeof(ProxyIsotope));    
     Get(i)->Relocate(vector_d[i]);
  }

  // copy the pointer to alias table pointers from the host to the device
  cudaMemcpy(d_fIsotopeVector, vector_d, vectorSize, cudaMemcpyHostToDevice);

  fData = d_fIsotopeVector;

  // copy the manager from host to device.
  bufferSize = vectorSize + 2*sizeof(size_t) + sizeof(bool);
  cudaMemcpy(devPtr, this, bufferSize, cudaMemcpyHostToDevice);

  // persistency
  fData = h_fIsotopeVector;

}

} // namespace geantx
