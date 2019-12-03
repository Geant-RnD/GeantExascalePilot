//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyEmModel.hpp
 * @brief The base class of EM models
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "Geant/track/TrackState.hpp"
#include "Geant/proxy/ProxyRandom.hpp"
#include "Geant/material/MaterialProperties.hpp"


namespace geantx {

//----------------------------------------------------------------------------//
// This is the base class for an EM model
//
template <class TEmModel>
class ProxyEmModel {

public:
  ProxyEmModel();
  ~ProxyEmModel()               = default;

  ProxyEmModel(const ProxyEmModel &) = default;
  ProxyEmModel(ProxyEmModel &&)      = default;

  ProxyEmModel &operator=(const ProxyEmModel &) = delete;
  ProxyEmModel &operator=(ProxyEmModel &&) = default;

public:

  void Initialization();

  void BuildAliasTable(bool atomicDependentModel = false) {}

  double MacroscopicCrossSection(const TrackState *track) 
  {  
    double xsec = 0.0;
    const double kinenergy = track->fPhysicsState.fEkin;

    if (kinenergy <= fLowEnergyLimit || kinenergy > fHighEnergyLimit) {
      return xsec;
    }

    const Material_t *mat = track->fMaterialState.fMaterial;
    const Vector_t<Element *> &theElements  = mat->GetElementVector();
    const double *theAtomicNumDensityVector = mat->GetMaterialProperties()->GetNumOfAtomsPerVolumeVect();

    for (size_t iel = 0; iel < theElements.size() ; ++iel) {
      xsec += theAtomicNumDensityVector[iel] * static_cast<TEmModel *>(this)->CrossSectionPerAtom(theElements[iel]->GetZ(), kinenergy);
    }

    return xsec;
  }

  int SampleSecondaries(TrackState *track) 
  {  
    return static_cast<TEmModel *>(this) -> SampleSecondaries(track);
  }

protected:

  bool fAtomicDependentModel;
  double fLowEnergyLimit;
  double fHighEnergyLimit;

  ProxyRandom *fRng = nullptr;
};

template <typename TEmModel>
ProxyEmModel<TEmModel>::ProxyEmModel() 
  : fAtomicDependentModel(false), 
    fLowEnergyLimit(100.0 * geantx::units::eV), 
    fHighEnergyLimit(100.0 * geantx::units::TeV)
{ 
  fRng = new ProxyRandom; 
}

} // namespace geantx
