//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/NistElementManager.hpp
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"

#include "Geant/proxy/ProxyElement.cuh"

#include <string>

#include <iostream>
#include <cassert>

namespace geantx {

struct NistElement {
  std::string fName;
  int fNumberOfIsotopes;
  int fFirstMassNumber;
  int fFirstIsotopeIndex;
  double fAtomicWeight;
};

struct NistIsotope {
  double fAtomicMass;
  double fComposition; 
};

class NistElementManager 
{
public:

  GEANT_HOST
  static NistElementManager *Instance();

  GEANT_HOST
  NistElementManager();

  GEANT_HOST
  ~NistElementManager() {}

  // NIST Elemenets

  GEANT_HOST
  inline int GetNumberOfElements() const { return fNumberOfElements; }  

  GEANT_HOST
  std::string GetElementName(int z) const; 

  GEANT_HOST
  int GetNumberOfIsotopes(int z) const; 

  GEANT_HOST
  int GetFirstMassNumber(int z) const; 

  GEANT_HOST
  double GetStandardAtomicWeight(int z) const;

  // NIST Isotopes

  GEANT_HOST
  inline int GetNumberOfIsotopes() const { return fNumberOfIsotopes; }  

  GEANT_HOST
  double GetRelativeAtomicMass(int z, int n) const;

  GEANT_HOST
  double GetIsotopicComposition(int z, int n) const;

  GEANT_HOST
  void Print(int z);

private:

  GEANT_HOST
  bool LoadData();

  GEANT_HOST
  bool CheckZ(int z) const;

  GEANT_HOST
  bool CheckZN(int z, int n) const;

  GEANT_HOST
  int GetIsotopeIndex(int z, int n) const;

private:
  static NistElementManager *fInstance;

  int fNumberOfElements;
  int fNumberOfIsotopes;
  NistElement* fElement;
  NistIsotope* fIsotope;
};

} // namespace geantx
