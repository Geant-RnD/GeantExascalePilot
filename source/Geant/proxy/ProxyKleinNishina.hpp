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
 * @file
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/SystemOfUnits.hpp"
#include "Geant/core/PhysicalConstants.hpp"
#include "Geant/proxy/ProxyEmModel.hpp"
#include <VecCore/VecCore>

namespace geantx {
class ProxyKleinNishina : public ProxyEmModel<ProxyKleinNishina> {

public:
  ProxyKleinNishina()  = default;
  ~ProxyKleinNishina() = default;

  //mandatory methods
  double CrossSectionPerAtom(int Z, double energy);

  int SampleSecondaries(int Z, double energy);

private:

  friend class ProxyEmModel<ProxyKleinNishina>;

};

// based on Geant4 processes/electromagnetic/standard/src/G4KleinNishinaModel.cc
double ProxyKleinNishina::CrossSectionPerAtom(int Z, double gammaEnergy)
{
  double xSection = 0.;

  const double dT0 = geantx::units::keV;
  const double barn = 1.0;

  double Z2 = Z * Z;
  double p1 =  2.7965e-1 +  1.9756e-5 * Z + -3.9178e-7 * Z2;
  double p2 = -1.8300e-1 + -1.0205e-2 * Z +  6.8241e-5 * Z2;
  double p3 =  6.7527    + -7.3913e-2 * Z +  6.0480e-5 * Z2;
  double p4 = -1.9798e+1 +  2.7079e-2 * Z +  3.0274e-4 * Z2;

  double T0 = (Z < 1.5) ? 40.0 * geantx::units::keV : 15.0 * geantx::units::keV;

  bool isLowEnergy = (gammaEnergy < T0);
  
  double X  = (isLowEnergy) ?  (T0 + dT0)/geantx::units::kElectronMassC2 :
      vecCore::math::Max(gammaEnergy, T0)/geantx::units::kElectronMassC2 ;
  double X2 = X * X;
  double sigma =
    p1 * vecCore::math::Log(1. + 2.*X)/X + (p2 + p3 * X + p4 * X2)/(1. + 20. * X + 230. * X2 + 440. * X2 * X);

  //  modification for low energy. (special case for Hydrogen)
  if (gammaEnergy < T0) {
    double c1 = -T0 * (Z*sigma*barn - xSection) / (xSection * dT0);
    double c2 = (Z > 1.5) ? 0.375 - 0.0556 * vecCore::math::Log(Z) : 0.150;
    double y = vecCore::math::Log(gammaEnergy / T0);
    xSection *= vecCore::math::Exp(-y * (c1 + c2 * y));
  }
  else {
    xSection = Z * sigma * barn;
  }
  
  return (xSection < 0.0) ? 0.0 : xSection;

}

// based on Geant4 processes/electromagnetic/standard/src/G4KleinNishinaModel.cc
int ProxyKleinNishina::SampleSecondaries(int Z, double gammaEnergy)
{
  int nsecondaries = 0;

  return nsecondaries;
}

} // namespace geantx
