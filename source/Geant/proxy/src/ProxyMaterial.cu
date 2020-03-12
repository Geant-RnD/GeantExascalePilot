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
#include "Geant/proxy/ProxyElementUtils.cuh"

namespace geantx {

GEANT_HOST
ProxyMaterial::ProxyMaterial(const char* name, double density, 
                             double z, double a, MaterialState state) 
  : fName(name), fState(state)
{

  fNumberOfElements = 1;
  fState = state;
  
  fDensity  = density; 

  AddElement(new ProxyElement("",z,a),1.0);

  ComputeDerivedQuantities();

  fState = (fState == kNullState && fDensity > clhep::kGasThreshold) ? kSolid : kGas;
  
  //add to the table
  StoreMaterial();
}

GEANT_HOST
ProxyMaterial::ProxyMaterial(const char* name, double density, 
                             int nElements, MaterialState state)
  : fName(name), fState(state)
{
  fNumberOfElements = 0;
  
  fDensity  = density; 
  fRadlen = 0;

  fIndex = -nElements; // use fIndex as a tempoary counter until this material is filled

  fAtomsVector = new int [nElements];
  fMassFractionVector = new double [nElements];
  fElementVector  = new ProxyElement * [nElements];

  for(int i = 0 ; i < nElements ; ++i) {
    fElementVector[i] = new ProxyElement;
    fAtomsVector[i] = 0;
    fMassFractionVector[i] = 1.0;
 }

  fState = (fState == kNullState && fDensity > clhep::kGasThreshold) ? kSolid : kGas;
  
  StoreMaterial();
}

GEANT_HOST
ProxyMaterial::~ProxyMaterial()
{
  delete fElementVector;
  delete fMassFractionVector;
  delete fNumberOfAtomsPerVolume;
}  

GEANT_HOST
void ProxyMaterial::StoreMaterial()
{
  //add this to the material table
  fIndex = ProxyMaterialTable::Instance()->size();
  ProxyMaterialTable::Instance()->push_back(this);
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

  cudaMemcpy(devPtr, this, MemorySize(), cudaMemcpyHostToDevice);

  // persistency
  fElementVector = h_fElementVector;
  fName = h_name;

#endif
}
  
GEANT_HOST
void ProxyMaterial::ComputeDerivedQuantities()
{
  double radinv = 0.0 ;

  for (size_t i=0 ; i < fNumberOfElements ; ++i) {

    double Zi = fElementVector[i]->GetZ();
    double Ai = fElementVector[i]->GetA();

    fNumberOfAtomsPerVolume[i] = clhep::Avogadro*fDensity*fMassFractionVector[i]/Ai;
    fAtomDensity += fNumberOfAtomsPerVolume[i];
    fElectronDensity += fNumberOfAtomsPerVolume[i]*Zi;

    radinv += fNumberOfAtomsPerVolume[i]*ProxyElementUtils::ComputeLradTsaiFactor(Zi);
  }

  fRadlen = (radinv <= 0.0 ? DBL_MAX : 1./radinv);
}

GEANT_HOST
void ProxyMaterial::AddElement(ProxyElement *element, int nAtoms)
{
  if(nAtoms <= 0) assert(0); 

  fElementVector[fNumberOfElements] = element;
  fAtomsVector[fNumberOfElements] = nAtoms;

  ++fNumberOfElements;
  ++fIndex;

   //filled by all proposed numbers of elements
  if(fIndex == 0) {
    double Amol = 0.;
    for (int i=0; i < fNumberOfElements; ++i) {
      fMassFractionVector[i] = fAtomsVector[i]*fElementVector[i]->GetA();
      Amol += fMassFractionVector[i];
    }

    for (int i=0; i < fNumberOfElements; ++i) {
      fMassFractionVector[i] /= Amol;
    }

    //calcuate derived quantities
    ComputeDerivedQuantities();
  }
}

GEANT_HOST
void ProxyMaterial::AddElement(ProxyElement *element, double fraction)
{
  if(fraction < 0.0 || fraction > 1.0) {
    assert(0); 
  }

  fElementVector[fNumberOfElements] = element;
  fMassFractionVector[fNumberOfElements] = fraction;

  ++fNumberOfElements;
  ++fIndex;

   //filled by all proposed numbers of elements
  if(fIndex == 0) {
    double Amol = 0.;
    for (int i=0; i < fNumberOfElements; ++i) {
      Amol += fMassFractionVector[i]*fElementVector[i]->GetA();
    }

    for (int i=0; i < fNumberOfElements; ++i) {
      fAtomsVector[i] = std::lrint(fMassFractionVector[i]*Amol/fElementVector[i]->GetA());
    }

    //calcuate derived quantities
    ComputeDerivedQuantities();
  }
}

} //namespace geantx
