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
 * @file Geant/proxy/src/ProxyElement.cu
 * @brief
 */
//===----------------------------------------------------------------------===//

#include <cassert>

#include "Geant/proxy/ProxyIsotope.cuh"
#include "Geant/proxy/ProxyElement.cuh"
#include "Geant/proxy/ProxyElementTable.cuh"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"

#include <iostream>

namespace geantx {

GEANT_HOST
ProxyElement::ProxyElement(const char* name, double zeff, double aeff)
  : fName(name), fZeff(zeff), fAeff(aeff), fNumberOfIsotopes(0),
    fWeights(nullptr), fIsotopeVector(nullptr)
{
  fNeff = fAeff/(clhep::g/clhep::mole);

  if(fNeff < 1.0) fNeff = 1.0;

  StoreElement();

}

GEANT_HOST
ProxyElement::ProxyElement(const char* name, int nIsotopes)
  : fName(name)
{
  fNumberOfIsotopes = 0;
  fIndex = -nIsotopes; // use fIndex as a tempoary counter until this element is filled

  fWeights = new double [nIsotopes];
  fIsotopeVector = new ProxyIsotope * [nIsotopes];

  for(int i = 0 ; i < nIsotopes ; ++i) {
    fIsotopeVector[i] = new ProxyIsotope;
    fWeights[i] = 1.0;
  }
}

GEANT_HOST
ProxyElement::~ProxyElement()
{
  delete fWeights;
  delete [] fIsotopeVector;
}

GEANT_HOST
void ProxyElement::StoreElement() {

  //add to the element table
  ProxyElementTable::Instance()->push_back(this);
  fIndex = ProxyElementTable::Instance()->size() - 1;
}

GEANT_HOST
void ProxyElement::AddIsotope(ProxyIsotope* isotope, double weight, bool store)
{
  //check Z if fNumberOfIsotopes > 0
  int iz = isotope->GetZ();
  
  if( fNumberOfIsotopes > 0) {
    assert(fZeff != double(iz));
  }
  else {
    fZeff = double(iz); 
  }
  
  fIsotopeVector[fNumberOfIsotopes] = isotope;
  fNumberOfIsotopes++;
  fIndex++;

  //filled by all proposed numbers of isotopes
  if(fIndex == 0) {
    double wtSum = 0.0;

    for (int i=0; i < fNumberOfIsotopes; ++i) {
      fAeff +=  fWeights[i] * fIsotopeVector[i]->GetA();
      wtSum +=  fWeights[i];
    }

    if(wtSum > 0.0) { fAeff  /= wtSum; }
    fNeff   = fAeff/(clhep::g/clhep::mole);

    if(wtSum != 1.0) {
      for(int i=0; i<fNumberOfIsotopes; ++i) { 
        fWeights[i] /= wtSum; 
      }
    }  

    // add to the proxy element table
    if(store) StoreElement();
  }
}

GEANT_HOST_DEVICE
size_t ProxyElement::MemorySize() {
  size_t bufferSize = sizeof(fName) + 2*sizeof(int) + 3*sizeof(double);
  if( fNumberOfIsotopes > 0 ) {
    for(int i = 0 ; i < fNumberOfIsotopes ; ++i ) {
      bufferSize += fIsotopeVector[i]->MemorySize();
    }
    bufferSize += fNumberOfIsotopes*sizeof(double);
  }
  else {
    bufferSize += sizeof(ProxyIsotope**);
    bufferSize += sizeof(double*);
  }
  return bufferSize;
}  

GEANT_HOST
void ProxyElement::Relocate(void* devPtr)
{
#ifdef GEANT_CUDA
  char* d_name;
  cudaMalloc((void **)&(d_name), sizeof(fName));
  cudaMemcpy(d_name, fName, sizeof(fName), cudaMemcpyHostToDevice);

  const char* h_name = fName;
  fName = d_name;

  //weight vector
  double *d_fWeights;
  double *h_fWeights = fWeights;

  cudaMalloc((void **)&(d_fWeights), sizeof(double) * fNumberOfIsotopes);
  cudaMemcpy(d_fWeights, fWeights, sizeof(double) * fNumberOfIsotopes,
	     cudaMemcpyHostToDevice);
  fWeights = d_fWeights;

  //isotope vector
  ProxyIsotope** h_fIsotopeVector = fIsotopeVector;

  if( fNumberOfIsotopes > 0 ) {

    // device pointers in device memory
    ProxyIsotope** d_fIsotopeVector;
    cudaMalloc((void **)&d_fIsotopeVector, fNumberOfIsotopes*sizeof(ProxyIsotope));
    
    // device pointers in host memory
    ProxyIsotope *vector_d[fNumberOfIsotopes];

    for(int i = 0 ; i < fNumberOfIsotopes ; ++i) {
      cudaMalloc((void**)&vector_d[i], sizeof(ProxyIsotope));
      fIsotopeVector[i]->Relocate(vector_d[i]);  
    }
    cudaMemcpy(d_fIsotopeVector, vector_d, sizeof(ProxyIsotope) *fNumberOfIsotopes, cudaMemcpyHostToDevice);

    fIsotopeVector = d_fIsotopeVector;
  }

  cudaMemcpy(devPtr, this, MemorySize(), cudaMemcpyHostToDevice);

  // persistency
  fIsotopeVector = h_fIsotopeVector;
  fName = h_name;
  fWeights = h_fWeights;
  
#endif
}

} // namespace geantx
