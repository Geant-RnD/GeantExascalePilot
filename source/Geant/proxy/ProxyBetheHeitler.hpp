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
 * @file Geant/proxy/ProxyBetheHeitler.hpp
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

namespace geantx {
class ProxyBetheHeitler : public ProxyEmModel<ProxyBetheHeitler> {

public:
  ProxyBetheHeitler() { fLowEnergyLimit = 2.0*geantx::units::kElectronMassC2;}
  ProxyBetheHeitler(const ProxyBetheHeitler &model) : ProxyEmModel<ProxyBetheHeitler>() { this->fRng = model.fRng; }
  ~ProxyBetheHeitler() = default;

  //mandatory methods for static polymorphism
  int SampleSecondaries(TrackState *track);

  //auxiliary methods
  double ScreenFunction1(double screenVariable) const;
  double ScreenFunction2(double screenVariable) const;

private:

  friend class ProxyEmModel<ProxyBetheHeitler>;

};

// based on Geant4 processes/electromagnetic/standard/src/G4BetheHeitlerModel.cc
int ProxyBetheHeitler::SampleSecondaries(TrackState *track)
{
  int nsecondaries = 0;

  double gammaEnergy = track->fPhysicsState.fEkin;

  double epsil0 = geantx::units::kElectronMassC2/gammaEnergy;
  if( epsil0 > 0.5) return nsecondaries;

  // select randomly one element constituing the material - input
  //  int index = track->fMaterialState.fMaterialId;
  int elementZ = 10; //@@@syjun: temporay

  const double Egsmall = 2. * geantx::units::MeV;  //@@@syjun: check unit

  double epsil;

  if (gammaEnergy < Egsmall) {
   // do it fast if gammaEnergy < Egsmall
    epsil = epsil0 + (0.5 - epsil0) * this->fRng->uniform();
  } else {
    // Extract Coulomb factor for this Element

    double logZ3 = vecCore::math::Log(1.0 * int(elementZ + 0.5)) / 3.0;
    double FZ    = 8. * logZ3; //(anElement->GetIonisation()->GetlogZ3());
    if (gammaEnergy > 50. * geantx::units::MeV) { //@@@syjun: check unit
      FZ += 8. * ComputeCoulombFactor(elementZ);
    }

    // limits of the screening variable
    double Z3        = vecCore::math::Pow(1.0 * int(elementZ + 0.5), 1 / 3.0);
    double screenfac = 136. * epsil0 / Z3; //(anElement->GetIonisation()->GetZ3());
    double screenmax = exp((42.24 - FZ) / 8.368) - 0.952;
    double screenmin = vecCore::math::Min(4. * screenfac, screenmax);

    // limits of the energy sampling
    double epsil1   = 0.5 - 0.5 * vecCore::math::Sqrt(1. - screenmin / screenmax);
    double epsilmin = vecCore::math::Max(epsil0, epsil1);
    double  epsilrange = 0.5 - epsilmin;

    //
    // sample the energy rate of the created electron (or positron)
    //
    double screenvar, greject;

    double F10    = ScreenFunction1(screenmin) - FZ;
    double F20    = ScreenFunction2(screenmin) - FZ;
    double NormF1 = vecCore::math::Max(F10 * epsilrange * epsilrange, 0.);
    double NormF2 = vecCore::math::Max(1.5 * F20, 0.);

    do {
      if (NormF1 / (NormF1 + NormF2) > this->fRng->uniform() ) {
        epsil     = 0.5 - epsilrange * vecCore::math::Pow(this->fRng->uniform(), 0.333333);
        screenvar = screenfac / (epsil * (1 - epsil));
        greject   = (ScreenFunction1(screenvar) - FZ) / F10;
      } else {
        epsil     = epsilmin + epsilrange * this->fRng->uniform();
        screenvar = screenfac / (epsil * (1 - epsil));
        greject   = (ScreenFunction2(screenvar) - FZ) / F20;
      }

    } while (greject < this->fRng->uniform());
  } //  end of epsil sampling
  
  double electTotEnergy = (this->fRng->uniform() > 0.5) ? (1. - epsil) * gammaEnergy : epsil*gammaEnergy;
  double positTotEnergy = gammaEnergy - electTotEnergy;

  // scattered electron (positron) angles. ( Z - axis along the parent photon)
  // derived from Tsai distribution (Rev Mod Phys 49,421(1977))
  // G4ModifiedTsai::SampleCosTheta, G4ModifiedTsai::SamplePairDirections

  double aa0 = -vecCore::math::Log(this->fRng->uniform()*this->fRng->uniform());
  double u = (0.25 > this->fRng->uniform()) ? aa0/0.625 : aa0/1.875;

  double phi = geantx::units::kTwoPi * this->fRng->uniform();
  double cosp = vecCore::math::Cos(phi);
  double sinp = vecCore::math::Sin(phi);

  //secondary electron 
  double cost = u * geantx::units::kElectronMassC2/ electTotEnergy; 
  double sint = vecCore::math::Sqrt((1. - cost)*(1. + cost));

  double xhat = sint*cosp;
  double yhat = sint*sinp;
  double zhat = cost;

  TrackState* electron = new TrackState;
  Math::RotateToLabFrame(xhat, yhat, zhat, track->fDir.x(), track->fDir.y(), track->fDir.z());
  ThreeVector electronDirection(xhat, yhat, zhat);

  electron->fPhysicsState.fEkin = electTotEnergy - geantx::units::kElectronMassC2;
  electron->fDir = electronDirection;
  ++nsecondaries;

  //secondary positron
  cost = u * geantx::units::kElectronMassC2/ positTotEnergy; 
  sint = vecCore::math::Sqrt((1. - cost)*(1. + cost));

  xhat = -sint*cosp;
  yhat = -sint*sinp;
  zhat = cost;

  TrackState* positron = new TrackState;
  Math::RotateToLabFrame(xhat, yhat, zhat, track->fDir.x(), track->fDir.y(), track->fDir.z());
  ThreeVector positronDirection(xhat, yhat, zhat);

  positron->fPhysicsState.fEkin = positTotEnergy - geantx::units::kElectronMassC2;
  positron->fDir = positronDirection;
  ++nsecondaries;

  //TODO: push this electron/positron pair to the global secondary container

  //update primary gamma
  track->fPhysicsState.fEkin = 0.0;
  track->fStatus = TrackStatus::Killed;

  return nsecondaries;
}

double ProxyBetheHeitler::ScreenFunction1(double screenVariable) const
{
  // compute the value of the screening function 3*PHI1 - PHI2
  double screenVal;

  if (screenVariable > 1.)
    screenVal = 42.24 - 8.368 * vecCore::math::Log(screenVariable + 0.952);
  else
    screenVal = 42.392 - screenVariable * (7.796 - 1.961 * screenVariable);

  return screenVal;
}

double ProxyBetheHeitler::ScreenFunction2(double screenVariable) const
{
  // compute the value of the screening function 1.5*PHI1 - 0.5*PHI2
  double screenVal;

  if (screenVariable > 1.)
    screenVal = 42.24 - 8.368 * vecCore::math::Log(screenVariable + 0.952);
  else
    screenVal = 41.405 - screenVariable * (5.828 - 0.8945 * screenVariable);

  return screenVal;
}


} // namespace geantx