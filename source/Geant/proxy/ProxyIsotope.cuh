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
 * @file Geant/proxy/ProxyIsotope.cuh
 * @brief a minimal implementation for Isotope
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/Config.hpp"

namespace geantx
{

class ProxyIsotope
{
public:
  GEANT_HOST
  ProxyIsotope() {}

  GEANT_HOST
  ProxyIsotope(const char* name, int z, int n, double a = 0.0, double frac = 0.0);

  GEANT_HOST
  ~ProxyIsotope() {}

  GEANT_HOST
  void Relocate(void* devPtr);

  GEANT_HOST_DEVICE
  inline const char* GetName() const { return fName; } 

  GEANT_HOST_DEVICE
  inline int GetZ() const { return fZ; } 

  GEANT_HOST_DEVICE
  inline int GetN() const { return fN; } 

  GEANT_HOST_DEVICE
  inline double GetA() const { return fA; } 

  GEANT_HOST_DEVICE
  inline unsigned int GetIndex() const { return fIndex; } 

  GEANT_HOST_DEVICE
  inline size_t MemorySize() { return sizeof(*this); }

  GEANT_HOST_DEVICE
  inline void Print();

private:

  const char*  fName;       // name of the isotope
  int          fZ;          // atomic number
  int          fN;          // number of nucleons
  double       fA;          // atomic mass of a mole
  double       fAbundance;  // fraction of natural abundance
  unsigned int fIndex;      // index in the Isotope table

};

GEANT_HOST_DEVICE
void ProxyIsotope::Print()
{
  printf("Isotope: Name= %s Z= %d N= %d A= %g fraction =%g Index=%d\n", 
          fName, fZ, fN, fA, fAbundance, fIndex);
}

} // namespace geantx

