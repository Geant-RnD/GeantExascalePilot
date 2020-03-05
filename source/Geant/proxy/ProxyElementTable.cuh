//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyElementTable.hpp
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyVector.cuh"
#include "Geant/proxy/ProxyMaterial.cuh"

namespace geantx {

class ProxyElementTable : public ProxyVector<ProxyElement*>
{

public:
  GEANT_HOST
  static ProxyElementTable *Instance();

  GEANT_HOST
  ProxyElementTable() : ProxyVector() {}

  GEANT_HOST
  ~ProxyElementTable() {}

  GEANT_HOST_DEVICE
  ProxyElementTable &operator=(ProxyElementTable const &other)
  {
    ProxyVector<ProxyElement*>::operator=(other);
    return *this;
  }

  GEANT_HOST
  void Relocate(void* devPtr);

  GEANT_HOST_DEVICE
  inline void Print();

  GEANT_HOST_DEVICE
  inline size_t Size() { return size(); } 

  GEANT_HOST_DEVICE
  inline size_t MemorySize();


private:
  static ProxyElementTable *fInstance;

};

GEANT_HOST_DEVICE
void ProxyElementTable::Print()
{
  printf("ProxyElementTable size = %d capacity=%d\n", size(), capacity());
  for (int i = 0 ; i < size() ; ++i) {
     Get(i)->Print();
  }
}

GEANT_HOST
size_t ProxyElementTable::MemorySize()
{
  //return size()*sizeof(ProxyElement) +  2*sizeof(size_t) + sizeof(bool); } 

  size_t buffer = 0; 
  for (int i = 0 ; i < size() ; ++i) {
     buffer += Get(i)->MemorySize();
  }
  buffer += 2*sizeof(size_t) + sizeof(bool);

  return buffer;
}



} // namespace geantx
