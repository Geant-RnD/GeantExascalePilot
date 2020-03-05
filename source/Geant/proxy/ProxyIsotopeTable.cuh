//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyIsotopeTable.hpp
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyVector.cuh"
#include "Geant/proxy/ProxyIsotope.cuh"

namespace geantx {

class ProxyIsotopeTable : public ProxyVector<ProxyIsotope*>
{
public:

  using ProxyVector<ProxyIsotope*>::ProxyVector;
  using typename ProxyVector<ProxyIsotope*>::iterator;
  using typename ProxyVector<ProxyIsotope*>::const_iterator;

  GEANT_HOST
  static ProxyIsotopeTable *Instance();

  GEANT_HOST
  ProxyIsotopeTable() {}

  GEANT_HOST
  ~ProxyIsotopeTable() {}

  GEANT_HOST_DEVICE
  ProxyIsotopeTable &operator=(ProxyIsotopeTable const &other)
  {
    ProxyVector<ProxyIsotope*>::operator=(other);
    return *this;
  }

  GEANT_HOST
  void Relocate(void* devPtr);

  GEANT_HOST_DEVICE
  inline void Print();

  GEANT_HOST_DEVICE
  inline size_t Size() { return size()*sizeof(ProxyIsotope) +  2*sizeof(size_t) + sizeof(bool); } 

private:
  static ProxyIsotopeTable *fInstance;

};

GEANT_HOST_DEVICE
void ProxyIsotopeTable::Print()
{
  printf("ProxyIsotopeTable size = %d capacity=%d\n", size(), capacity());
  for (int i = 0 ; i < size() ; ++i) {
     Get(i)->Print();
  }
}

} // namespace geantx
