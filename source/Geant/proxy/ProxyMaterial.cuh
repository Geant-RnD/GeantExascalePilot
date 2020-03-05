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
 * @file Geant/proxy/ProxyMaterial.cuh
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/proxy/ProxyElement.cuh"

namespace geantx
{

enum MaterialState { kNullState = 0, kSolid, kLiquid, kGas};

class ProxyMaterial 
{
public:

  GEANT_HOST
  ProxyMaterial() {}

  GEANT_HOST
  ProxyMaterial(const char* name, double density, double z, double a,
                MaterialState state = kNullState);

  GEANT_HOST
  ProxyMaterial(const char* name, double density, int nElements,
                MaterialState state = kNullState);

  GEANT_HOST_DEVICE
  ~ProxyMaterial();
  
  // accessors 

  GEANT_HOST_DEVICE
  inline const char* GetName() { return fName; } 

  GEANT_HOST_DEVICE
  inline size_t GetIndex() const { return fIndex; }

  GEANT_HOST_DEVICE
  inline size_t GetNumberOfElements() const { return fNumberOfElements; }

  GEANT_HOST_DEVICE
  inline double GetDensity() const { return fDensity; }
  
  GEANT_HOST_DEVICE
  inline double GetRadlen() const { return fRadlen; }

  GEANT_HOST_DEVICE
  inline ProxyElement** GetElementVector() const { return fElementVector; }

  GEANT_HOST_DEVICE
  inline ProxyElement* GetElement(int index) const { return fElementVector[index]; }

  // member methods

  GEANT_HOST
  void ComputeRadiationLength();
  
  GEANT_HOST
  void AddElement(ProxyElement* element, double fraction);

  GEANT_HOST
  void Relocate(void* devicePtr);
  
//  GEANT_HOST
//  inline void AddToMaterialTable(void* this) { ; /* dummy for now */ }

  GEANT_HOST_DEVICE
  inline size_t MemorySize();

  GEANT_HOST_DEVICE
  inline void Print();

private:

  const char*    fName;              // name of the element
  size_t         fIndex;             // index in the material table
  MaterialState  fState;

  double         fDensity;       
  double         fRadlen;

  int            fNumberOfElements;
  ProxyElement** fElementVector;

};

GEANT_HOST_DEVICE
size_t ProxyMaterial::MemorySize() {

  size_t bufferSize = sizeof(fName)+ sizeof(int) + sizeof(size_t) + sizeof(MaterialState) +2*sizeof(double);

  if( fNumberOfElements > 0 ) {
    for(int i = 0 ; i < fNumberOfElements ; ++i ) {
      bufferSize += fElementVector[i]->MemorySize();
    }
  }
  else {
    bufferSize += sizeof(ProxyElement**);
  }
      
  return bufferSize;
}  


GEANT_HOST_DEVICE 
void ProxyMaterial::Print()
{
  printf("ProxyMaterial %s Deinsity= %f Index=%d\n",fName,fDensity,fIndex);

  if(fNumberOfElements > 0) {
    printf("  Number Of Element = %d\n",fNumberOfElements);
    for(int i = 0 ; i < fNumberOfElements ; ++i) {
      printf("  Element[%d]\n",i);
      GetElement(i)->Print();
    }
  }
}


} // namespace geantx
