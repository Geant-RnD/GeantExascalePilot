//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/processes/EmModel.hpp
 * @brief The base class of EM models
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

namespace geantx {

//----------------------------------------------------------------------------//
// This is the base class for an EM model
//
template <class TEmModel>
class ProxyEmModel {

public:
  ProxyEmModel(const std::string &name) {}
  ProxyEmModel()                = default;
  ~ProxyEmModel()               = default;
  ProxyEmModel(const ProxyEmModel &) = default;
  ProxyEmModel(ProxyEmModel &&)      = default;

  ProxyEmModel &operator=(const ProxyEmModel &) = delete;
  ProxyEmModel &operator=(ProxyEmModel &&) = default;

public:

  void Initialization();

  void BuildAliasTable(bool atomicDependentModel = false) {}

  double CrossSectionPerAtom(int Z, double energy) 
  {  
    return static_cast<TEmModel *>(this) -> CrossSectionPerAtom(Z, energy);
  }

  int SampleSecondaries(int Z, double energy) 
  {  
    return static_cast<TEmModel *>(this) -> SampleSecondaries(Z, energy);
  }

protected:
  bool fAtomicDependentModel;
  double fLowEnergyLimit;
  double fHighEnergyLimit;
};

} // namespace geantx
