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
 * @file Geant/proxy/ProxyElement.cuh
 * @brief a minimal implementation for Element
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyIsotope.cuh"

namespace geantx
{

class ProxyElement
{
public:

  GEANT_HOST
  ProxyElement() {}

  GEANT_HOST
  ProxyElement(const char* name, double zeff, double aeff);

  GEANT_HOST
  ProxyElement(const char* name, int nIsotopes);

  GEANT_HOST
  ~ProxyElement();

  GEANT_HOST
  void StoreElement();

  GEANT_HOST
  void Relocate(void* devPtr);

  GEANT_HOST
  void AddIsotope(ProxyIsotope* isotope, double weight, bool store=true);

  GEANT_HOST_DEVICE
  inline const char* GetName() { return fName; } 

  GEANT_HOST_DEVICE
  inline double GetZ() { return fZeff; } 

  GEANT_HOST_DEVICE
  inline double GetN() { return fNeff; } 

  GEANT_HOST_DEVICE
  inline double GetA() { return fAeff; } 

  GEANT_HOST_DEVICE
  inline int GetIndex() { return fIndex; } 

  GEANT_HOST_DEVICE
  inline int GetNumberOfIsotopes() { return fNumberOfIsotopes; } 

  GEANT_HOST_DEVICE
  inline  ProxyIsotope** GetIsotopeVector() { return fIsotopeVector; } 

  GEANT_HOST_DEVICE
  inline  ProxyIsotope* GetIsotope(int idx) { return fIsotopeVector[idx]; } 

  GEANT_HOST_DEVICE
  size_t MemorySize();

  GEANT_HOST_DEVICE
  inline void Print();

private:

  const char* fName;               // name of the element
  double fZeff;                    // effective atomic number
  double fNeff;                    // effective number of nucleons
  double fAeff;                    // effective mass of a mole

  int fNumberOfIsotopes;           // number of isotopes of the element
  double* fWeights;                // pointer to an array of weights 
  ProxyIsotope** fIsotopeVector;   // pointer to an array of isotopes 

  int fIndex;                      // index in the Element Table
};

GEANT_HOST_DEVICE
void ProxyElement::Print()
{
  printf("ProxyElement: %s Z= %g N= %g A= %g Index=%d\n",fName,fZeff,fNeff,fAeff,fIndex);

  if(fNumberOfIsotopes > 0) {
    printf("Number Of Isotopes = %d\n",fNumberOfIsotopes);
    for(int i = 0 ; i < fNumberOfIsotopes ; ++i) {
      GetIsotope(i)->Print();
    }
  }
}

} // namespace geantx

