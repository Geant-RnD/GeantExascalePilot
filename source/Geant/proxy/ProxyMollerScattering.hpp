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
 * @file Geant/proxy/ProxyMollerScattering.hpp 
 * @brief the Moller model of the electron ionization (e-e- scattering)
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
class ProxyMollerScattering : public ProxyEmModel<ProxyMollerScattering> {

public:
  ProxyMollerScattering() { fLowEnergyLimit = 100.0 * geantx::units::eV; }
  ProxyMollerScattering(const ProxyMollerScattering &model) : ProxyEmModel<ProxyMollerScattering>() { this->fRng = model.fRng; }
  ~ProxyMollerScattering() = default;

  //mandatory methods
  double CrossSectionPerAtom(double Z, double kineticEnergy);
  int SampleSecondaries(TrackState *track);

  //auxiliary methods

private:

  friend class ProxyEmModel<ProxyMollerScattering>;

};

// based on Geant4 processes/electromagnetic/standard/src/G4MollerBhabhaModel
double ProxyMollerScattering::CrossSectionPerAtom(double Z, double kineticEnergy)
{
  double xsec = 0.0;

  //TODO: MaterialCuts by an argument
  const double cutEnergy = 10.0 * geantx::units::eV; // matcut->GetProductionCutsInEnergy()[1];

  double tmax = Math::Min(fHighEnergyLimit, 0.5*kineticEnergy);

  if(cutEnergy > tmax) return xsec;

  double xmin  = cutEnergy/kineticEnergy;
  double xmax  = tmax/kineticEnergy;
  double tau   = kineticEnergy/geantx::units::kElectronMassC2;
  double gam   = tau + 1.0;
  double gamma2= gam*gam;
  double beta2 = tau*(tau + 2)/gamma2;
  
  //Moller (e-e-) scattering
  double gg = (2.0*gam - 1.0)/gamma2;
  xsec = ((xmax - xmin)*(1.0 - gg + 1.0/(xmin*xmax)
			  + 1.0/((1.0-xmin)*(1.0 - xmax)))
	- gg*Math::Log( xmax*(1.0 - xmin)/(xmin*(1.0 - xmax)) ) ) / beta2;
  
  xsec *= geantx::units::kTwopi_mc2_rcl2/kineticEnergy;
  
  return Z * xsec;
}

// based on Geant4 processes/electromagnetic/standard/src/G4MollerBhabhaModel
int ProxyMollerScattering::SampleSecondaries(TrackState *track)
{
  int nsecondaries = 0;
  double kineticEnergy = track->fPhysicsState.fEkin;

  //cut energy
  double maxEnergy = 1.0 * geantx::units::TeV; //temp
  double cutEnergy = 1.0 * geantx::units::keV; //temp

  double tmin = cutEnergy;  
  double tmax = 0.5*kineticEnergy; 

  if(maxEnergy < tmax) { tmax = maxEnergy; }
  if(tmin >= tmax) { return nsecondaries; }

  double energy = kineticEnergy + geantx::units::kElectronMassC2;
  double xmin   = tmin/kineticEnergy;
  double xmax   = tmax/kineticEnergy;
  double gam    = energy/geantx::units::kElectronMassC2;
  double gamma2 = gam*gam;
  double x, z, grej;

  //Moller (e-e-) scattering

  double gg = (2.0*gam - 1.0)/gamma2;
  double y = 1.0 - xmax;
  grej = 1.0 - gg*xmax + xmax*xmax*(1.0 - gg + (1.0 - gg*y)/(y*y));
  
  do {
    double rndm = this->fRng->uniform();
    x = xmin*xmax/(xmin*(1.0 - rndm) + xmax*rndm);
    y = 1.0 - x;
    z = 1.0 - gg*x + x*x*(1.0 - gg + (1.0 - gg*y)/(y*y));
  } while(grej * this->fRng->uniform() > z);
  
  double deltaKinEnergy = x * kineticEnergy;
  
  //angle of the scatterred electron
  double totalMomentum = Math::Sqrt(kineticEnergy  * (kineticEnergy + 2.0* geantx::units::kElectronMassC2));
  double deltaMomentum = Math::Sqrt(deltaKinEnergy * (deltaKinEnergy + 2.0* geantx::units::kElectronMassC2));
  double cost =  deltaKinEnergy * (energy +  geantx::units::kElectronMassC2) / (deltaMomentum * totalMomentum);
  if(cost > 1.0) { cost = 1.0; }
  double sint = Math::Sqrt((1.0 - cost)*(1. + cost));

  double phi = geantx::units::kTwoPi * this->fRng->uniform();

  double xhat = sint*Math::Cos(phi);
  double yhat = sint*Math::Sin(phi);
  double zhat = cost;
  
  Math::RotateToLabFrame(xhat, yhat, zhat, track->fDir.x(), track->fDir.y(), track->fDir.z());
  ThreeVector deltaDirection(xhat, yhat, zhat);
  
  //create a delta ray (electron)
  TrackState* electron = new TrackState;
  electron->fDir = deltaDirection;
  electron->fPhysicsState.fEkin = deltaKinEnergy;

  //TODO: push this secondary to the global secondary container
  ++nsecondaries;

  //update the primary
  
  track->fPhysicsState.fEkin -= deltaKinEnergy;
  track->fDir = (totalMomentum*track->fDir - deltaMomentum*deltaDirection).Unit();

  return nsecondaries;
}

} // namespace geantx
