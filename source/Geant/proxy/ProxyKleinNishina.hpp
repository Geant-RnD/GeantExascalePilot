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
 * @file Geant/proxy/ProxyKleinNishina.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/SystemOfUnits.hpp"
#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/proxy/ProxyEmModel.hpp"
#include <VecCore/VecCore>

#include "Geant/core/math_wrappers.hpp"
//#include "Geant/material/MaterialProperties.hpp"

namespace geantx {
class ProxyKleinNishina : public ProxyEmModel<ProxyKleinNishina> {

public:
  ProxyKleinNishina() { fLowEnergyLimit = 100.0 * geantx::units::eV;}
  ProxyKleinNishina(const ProxyKleinNishina &model) : ProxyEmModel<ProxyKleinNishina>() { this->fRng = model.fRng; }
  ~ProxyKleinNishina() = default;

  //mandatory methods
  double CrossSectionPerAtom(double Z, double energy);
  int SampleSecondaries(TrackState *track);

  //auxiliary methods

private:

  friend class ProxyEmModel<ProxyKleinNishina>;

};

// based on Geant4 processes/electromagnetic/standard/src/G4KleinNishinaModel.cc
double ProxyKleinNishina::CrossSectionPerAtom(double Z, double gammaEnergy)
{
  double xSection = 0.;

  double Z2 = Z * Z;
  double p1 =  2.7965e-1 +  1.9756e-5 * Z + -3.9178e-7 * Z2;
  double p2 = -1.8300e-1 + -1.0205e-2 * Z +  6.8241e-5 * Z2;
  double p3 =  6.7527    + -7.3913e-2 * Z +  6.0480e-5 * Z2;
  double p4 = -1.9798e+1 +  2.7079e-2 * Z +  3.0274e-4 * Z2;

  double T0 = (Z < 1.5) ? 40.0 * geantx::units::keV : 15.0 * geantx::units::keV;

  double X  = Math::Max(gammaEnergy, T0)/geantx::units::kElectronMassC2 ;
  double X2 = X*X;

  xSection = p1 * Math::Log(1. + 2.*X)/X + (p2 + p3 * X + p4 * X2)/(1. + 20. * X + 230. * X2 + 440. * X2 * X);

  //  modification for low energy. (special case for Hydrogen)
  if (gammaEnergy < T0) {
    const double dT0 = geantx::units::keV;
    X  = (T0 + dT0)/geantx::units::kElectronMassC2;
    X2 = X*X;
    double sigma = p1 * Math::Log(1. + 2.*X)/X + (p2 + p3 * X + p4 * X2)/(1. + 20. * X + 230. * X2 + 440. * X2 * X);
    double c1 = -T0 * (sigma - xSection) / (xSection * dT0);
    double c2 = (Z > 1.5) ? 0.375 - 0.0556 * Math::Log(Z) : 0.150;
    double y = Math::Log(gammaEnergy / T0);
    xSection *= Math::Exp(-y * (c1 + c2 * y));
  }
  
  return Z*xSection*geantx::units::barn;
}

// based on Geant4 processes/electromagnetic/standard/src/G4KleinNishinaModel.cc
int ProxyKleinNishina::SampleSecondaries(TrackState *track)
{
  int nsecondaries = 0;
  double gammaEnergy0 = track->fPhysicsState.fEkin;

  if( gammaEnergy0 < fLowEnergyLimit) return nsecondaries;
  ThreeVector gammaDirection0 = track->fDir;

  double epsilon, epsilonsq, onecost, sint2, greject;

  double E0_m = gammaEnergy0/geantx::units::kElectronMassC2;
  
  double eps0 = 1. / (1. + 2. * E0_m);
  double epsilon0sq = eps0 * eps0;
  double alpha1 = -vecCore::math::Log(eps0);
  double alpha2 = 0.5 * (1. - epsilon0sq);

  do {
    if (alpha1 / (alpha1 + alpha2) > this->fRng->uniform()) {
      epsilon = vecCore::math::Exp(-alpha1 * this->fRng->uniform());
      epsilonsq = epsilon * epsilon;
    }
    else {
      epsilonsq = epsilon0sq + (1. - epsilon0sq) * this->fRng->uniform();
      epsilon = vecCore::math::Sqrt(epsilonsq);
    }

    onecost = (1. - epsilon) / (epsilon * E0_m);
    sint2 = onecost * (2. - onecost);
    greject = 1. - epsilon * sint2 / (1. + epsilonsq);

  } while (greject < this->fRng->uniform());

  //update kinematics for scattered gamma
  double sinTheta = (sint2 < 0.0) ? 0.0 : vecCore::math::Sqrt(sint2);
  double Phi     = geantx::units::kTwoPi * this->fRng->uniform();
  double gammaEnergy = epsilon * gammaEnergy0;

  ThreeVector gammaDirection(sinTheta*cos(Phi), sinTheta*sin(Phi), 1. - onecost);
  Math::RotateToLabFrame(gammaDirection.x(), gammaDirection.y(), gammaDirection.z(), 
			 track->fDir.x(), track->fDir.y(), track->fDir.z());

  if(gammaEnergy > fLowEnergyLimit) {
    track->fDir = gammaDirection;
    track->fPhysicsState.fEkin = gammaEnergy;
  }
  else {
    track->fStatus = TrackStatus::Killed;
    track->fPhysicsState.fEdep += gammaEnergy;  
  }

  //secondary electron
  double eKinEnergy = gammaEnergy0 - gammaEnergy;

  if(gammaEnergy > fLowEnergyLimit) {
    ThreeVector eDirection = gammaEnergy0*gammaDirection0 - gammaEnergy*gammaDirection;
    eDirection = eDirection.Unit();
    TrackState* electron = new TrackState;

    //TODO: push this secondary to the global secondary container
    ++nsecondaries;
  }
  else {
    track->fPhysicsState.fEdep += eKinEnergy;  
  }

  return nsecondaries;
}

} // namespace geantx
