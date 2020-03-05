//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyMaterialTable.hpp
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyVector.cuh"
#include "Geant/proxy/ProxyMaterial.cuh"

namespace geantx {

class ProxyMaterialTable : public ProxyVector<ProxyMaterial*>
{

public:
  GEANT_HOST
  static ProxyMaterialTable *Instance();

  GEANT_HOST
  ProxyMaterialTable() {}

  GEANT_HOST
  ~ProxyMaterialTable() {}

  GEANT_HOST_DEVICE
  ProxyMaterialTable &operator=(ProxyMaterialTable const &other)
  {
    ProxyVector<ProxyMaterial*>::operator=(other);
    return *this;
  }

  GEANT_HOST
  void Relocate(void* devPtr);

  GEANT_HOST_DEVICE
  inline void Print();

  GEANT_HOST_DEVICE
  inline size_t Size() { return size(); } 

  GEANT_HOST
  inline size_t MemorySize();

private:
  static ProxyMaterialTable *fInstance;

};

GEANT_HOST_DEVICE
void ProxyMaterialTable::Print()
{
  printf("ProxyMaterialTable size = %d capacity=%d\n", size(), capacity());
  for (int i = 0 ; i < size() ; ++i) {
     Get(i)->Print();
  }
}

GEANT_HOST
size_t ProxyMaterialTable::MemorySize()
{
  size_t buffer = 0; 
  for (int i = 0 ; i < size() ; ++i) {
     buffer += Get(i)->MemorySize();
  }
  buffer += 2*sizeof(size_t) + sizeof(bool);

  return buffer;
}


} // namespace geantx
