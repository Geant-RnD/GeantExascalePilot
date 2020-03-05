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

#include "Geant/proxy/ProxyMaterial.cuh"
#include "Geant/proxy/ProxyMaterialTable.cuh"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"

namespace geantx {

GEANT_HOST
ProxyMaterial::ProxyMaterial(const char* name, double density, double z, double a, MaterialState state)
{

  fName = name;
  fNumberOfElements = 1;
  fState = state;
  
  fDensity  = density; 

  //  ProxyElement* aElement = new ProxyElement(z,a);
  AddElement(new ProxyElement("",z,a),1.0);

  ComputeRadiationLength();

  fState = (fState == kNullState && fDensity > clhep::kGasThreshold) ? kSolid : kGas;
  
  //add to the table
  fIndex = ProxyMaterialTable::Instance()->size();
  ProxyMaterialTable::Instance()->push_back(this);
}

GEANT_HOST
ProxyMaterial::ProxyMaterial(const char* name, double density, int nElements,
			   MaterialState state)
{
  fName = name;
  fNumberOfElements = 0;
  fState = state;
  
  fDensity  = density; 
  fRadlen = 0;

  fElementVector  = new ProxyElement * [nElements];
  for(int i = 0 ; i < nElements ; ++i) {
    fElementVector[i] = new ProxyElement;
  }

  fState = (fState == kNullState && fDensity > clhep::kGasThreshold) ? kSolid : kGas;
  
  //add to the table
  fIndex = ProxyMaterialTable::Instance()->size();
  ProxyMaterialTable::Instance()->push_back(this);

}

GEANT_HOST
ProxyMaterial::~ProxyMaterial()
{
  delete fElementVector;
}  

GEANT_HOST
void ProxyMaterial::Relocate(void* devPtr)
{
#ifdef GEANT_CUDA
  char* d_name;
  cudaMalloc((void **)&(d_name), sizeof(fName));
  cudaMemcpy(d_name, fName, sizeof(fName), cudaMemcpyHostToDevice);

  const char* h_name = fName;
  fName = d_name;

  ProxyElement** h_fElementVector = fElementVector;

  size_t bufferSize = 0;
  size_t vectorSize = 0;

  if( fNumberOfElements > 0 ) {

    // device pointers in host memory
    ProxyElement *vector_d[fNumberOfElements];

    for(int i = 0 ; i < fNumberOfElements ; ++i) {
      vectorSize += fElementVector[i]->MemorySize(); 
      cudaMalloc((void**)&(vector_d[i]), fElementVector[i]->MemorySize());
      fElementVector[i]->Relocate(vector_d[i]);  
    }

    // device pointers in device memory
    ProxyElement** d_fElementVector;
    cudaMalloc((void **)&d_fElementVector, vectorSize);

    cudaMemcpy(d_fElementVector, vector_d, vectorSize, cudaMemcpyHostToDevice);

    fElementVector = d_fElementVector;
  }
  else {
    vectorSize = sizeof(ProxyElement**);
  }

  bufferSize = vectorSize + sizeof(int) + sizeof(size_t) + sizeof(fName)+sizeof(MaterialState) + 2*sizeof(double);

  cudaMemcpy(devPtr, this, bufferSize, cudaMemcpyHostToDevice);

  // persistency
  fElementVector = h_fElementVector;
  fName = h_name;

#endif
}

  
GEANT_HOST
void ProxyMaterial::ComputeRadiationLength()
{
  double radinv = 0.0 ;
  for (size_t i=0;i<fNumberOfElements;++i) {
    //    radinv += VecNbOfAtomsPerVolume[i]*GPElement_GetfRadTsai(&(fElementVector[i]));
    ;
  }
  fRadlen = (radinv <= 0.0 ? DBL_MAX : 1./radinv);
}

GEANT_HOST
void ProxyMaterial::AddElement(ProxyElement *element, double fraction)
{
  if(fraction < 0.0 || fraction > 1.0) {
    assert(0); 
  }

//  double Zi, Ai;

  fElementVector[fNumberOfElements] = element;

  //  fMassFractionVector[fNumberOfElements] = fraction;

//  Zi = element->GetZ();
//  Ai = element->GetA();

//  fZ += Zi*fraction;
//  fA += Ai*fraction;

  /*
  VecNbOfAtomsPerVolume[fNumberOfElements] = 
    Avogadro*fDensity*fMassFractionVector[fNumberOfElements]/Ai;
  TotNbOfAtomsPerVolume += 
    VecNbOfAtomsPerVolume[fNumberOfElements];
  TotNbOfElectPerVolume += 
    VecNbOfAtomsPerVolume[fNumberOfElements]*Zi;
  */
  ++(fNumberOfElements);


  //update radiation length
  ComputeRadiationLength();
}

} //namespace geantx
