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
 * @file Geant/proxy/ProxySauterGavrila.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/proxy/ProxySystemOfUnits.hpp"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"
#include "Geant/proxy/ProxySandiaData.hpp"
#include "Geant/proxy/ProxyEmModel.hpp"
#include <VecCore/VecCore>

#include "Geant/core/math_wrappers.hpp"

namespace geantx {
class ProxySauterGavrila : public ProxyEmModel<ProxySauterGavrila> {

public:
  GEANT_HOST
  ProxySauterGavrila();

  GEANT_HOST_DEVICE
  ProxySauterGavrila(int tid) : ProxyEmModel<ProxySauterGavrila>(tid) {}

  GEANT_HOST
  ProxySauterGavrila(const ProxySauterGavrila &model) 
   : ProxyEmModel<ProxySauterGavrila>() { this->fRng = model.fRng; }

  GEANT_HOST_DEVICE
  ~ProxySauterGavrila() {}

  //mandatory methods for static polymorphism
  GEANT_HOST_DEVICE
  int SampleSecondaries(TrackState *track);

  //auxiliary methods
  GEANT_HOST_DEVICE
  double  GetPhotoElectronEnergy(int Z, double gammaEnergy) const;

private:

  friend class ProxyEmModel<ProxySauterGavrila>;

};

GEANT_HOST
ProxySauterGavrila::ProxySauterGavrila() 
{ 
  fLowEnergyLimit = 1.0*geantx::clhep::eV; 
  fHighEnergyLimit = 100.0*geantx::clhep::MeV; 
}

// based on Geant4 processes/electromagnetic/standard/src/G4PEEffectFluoModel..cc
GEANT_HOST_DEVICE
int ProxySauterGavrila::SampleSecondaries(TrackState *track)
{
  int nsecondaries = 0;

  double gammaEnergy = track->fPhysicsState.fEkin;

  // use the scalar implementation which is equivalent to Geant4
  int Z = 10; //@@@syjun: connect to material

  double electronEnergy = GetPhotoElectronEnergy(Z, gammaEnergy);

  // sample angular direction according to SauterGavrilaAngularDistribution

  double tau = gammaEnergy / clhep::electron_mass_c2;

  double cost = -1.0;

  if (tau > 50.0 ) { // taulimit = 50.0;
    cost = 1.0; // set to the primary direction
  }
  else {
    // algorithm according Penelope 2008 manual and
    // F.Sauter Ann. Physik 9, 217(1931); 11, 454(1931).

    double gamma = tau + 1.;
    double beta = vecCore::math::Sqrt(tau * (tau + 2.)) / gamma;
    double A = (1 - beta) / beta;
    double Ap2 = A + 2.;
    double B = 0.5 * beta * gamma * (gamma - 1.) * (gamma - 2.);
    double grej = 2. * (1. + A * B) / A;
    double z, g;
    do {
      double q = this->fRng->uniform();
      z = 2 * A * (2 * q + Ap2 * vecCore::math::Sqrt(q)) / (Ap2 * Ap2 - 4 * q);
      g = (2 - z) * (1.0 / (A + z) + B);

    } while (g < this->fRng->uniform() * grej);

    cost = 1 - z;
  }

  double sint = vecCore::math::Sqrt((1 + cost) * (1 - cost));
  double phi = clhep::twopi * this->fRng->uniform();

  double xhat = sint*vecCore::math::Cos(phi);
  double yhat = sint*vecCore::math::Sin(phi);
  double zhat = cost;

  //photo electron 
  TrackState electron;
  Math::RotateToLabFrame(xhat, yhat, zhat, track->fDir.x(), track->fDir.y(), track->fDir.z());
  ThreeVector electronDirection(xhat, yhat, zhat);

  electron.fPhysicsState.fEkin = electronEnergy;
  electron.fDir = electronDirection;
  ++nsecondaries;

  //update primary gamma
  track->fPhysicsState.fEkin = 0.0;
  track->fStatus = TrackStatus::Killed;

  return nsecondaries;
}

GEANT_HOST_DEVICE
double ProxySauterGavrila::GetPhotoElectronEnergy(int Z, double energy) const
{
  assert(Z > 0 && Z <= 100);

  int nShells = sandia::fNumberOfShells[Z];

  int i = 0;
  while (i < nShells && energy >= sandia::fBindingEnergies[sandia::fIndexOfShells[Z] + i] * geantx::clhep::eV)
    i++;

  return i <= nShells ? energy - sandia::fBindingEnergies[sandia::fIndexOfShells[Z] + i] * geantx::clhep::eV : 0.0;

}

} // namespace geantx
