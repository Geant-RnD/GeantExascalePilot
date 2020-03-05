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
 * @file Geant/proxy/src/ProxyIsotope.cu
 * @brief
 */
//===----------------------------------------------------------------------===//

#include "stdio.h"
#include "Geant/proxy/ProxyIsotope.cuh"
#include "Geant/proxy/ProxyIsotopeTable.cuh"

namespace geantx {

GEANT_HOST
ProxyIsotope::ProxyIsotope(const char *name, int z, int n, double a, double frac)
  : fName(name), fZ(z), fN(n), fA(a), fAbundance(frac) 
{
	    
  assert(z>0 && z <= n);

  if(a<=0.0) {
//    fA = NistElement
  }

  //add to the global Isotope table
  fIndex = ProxyIsotopeTable::Instance()->size();
  ProxyIsotopeTable::Instance()->push_back(this);
}

GEANT_HOST
void ProxyIsotope::Relocate(void* devPtr)
{
#ifdef GEANT_CUDA
  char* d_name;
  cudaMalloc((void **)&(d_name), sizeof(fName));

  cudaMemcpy(d_name, fName, sizeof(fName), cudaMemcpyHostToDevice);

  const char* h_name = fName;  
  fName = d_name;
  cudaMemcpy(devPtr, this, sizeof(*this), cudaMemcpyHostToDevice);

  // persistency on CPU
  fName = h_name;

#endif
}

} // namespace geantx
