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
#include "Geant/proxy/ProxySystemOfUnits.hpp"
#include "Geant/material/MaterialProperties.hpp"

namespace geantx {

//----------------------------------------------------------------------------//
// This is the base class for an EM model
//
template <class TEmModel>
class ProxyEmModel {

public:

  GEANT_HOST
  ProxyEmModel();

  GEANT_HOST_DEVICE
  ProxyEmModel(int tid) : fThreadId(tid) {}

  GEANT_HOST_DEVICE
  ~ProxyEmModel() {}

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

  // accessor
  GEANT_HOST_DEVICE
  inline double GetLowEnergyLimit() {return fLowEnergyLimit; };

  GEANT_HOST_DEVICE
  inline double GetHighEnergyLimit() {return fHighEnergyLimit; };

  GEANT_HOST_DEVICE
  double ComputeCoulombFactor(double Zeff);

protected:

  int fThreadId;
  bool fAtomicDependentModel;
  double fLowEnergyLimit;
  double fHighEnergyLimit;

  ProxyRandom *fRng = nullptr;
};

template <typename TEmModel>
ProxyEmModel<TEmModel>::ProxyEmModel() 
  : fThreadId(-1),
    fAtomicDependentModel(false), 
    fLowEnergyLimit(100.0 * clhep::eV), 
    fHighEnergyLimit(100.0 * clhep::TeV)
{ 
  fRng = new ProxyRandom; 
}

template <typename TEmModel>
double ProxyEmModel<TEmModel>::ComputeCoulombFactor(double Zeff) 
{
  // Compute Coulomb correction factor (Phys Rev. D50 3-1 (1994) page 1254)

  double k1 = 0.0083, k2 = 0.20206, k3 = 0.0020, k4 = 0.0369;
  double fine_structure_const = (1.0 / 137); // check unit

  double az1 = fine_structure_const * Zeff;
  double az2 = az1 * az1;
  double az4 = az2 * az2;

  double coulombFactor = (k1 * az4 + k2 + 1. / (1. + az2)) * az2 - (k3 * az4 + k4) * az4;
  return coulombFactor;
}

} // namespace geantx
