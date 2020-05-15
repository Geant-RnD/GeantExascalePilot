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

#include "Geant/proxy/ProxySystemOfUnits.hpp"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"
#include "Geant/proxy/ProxyEmModel.hpp"
#include <VecCore/VecCore>

#include "Geant/core/math_wrappers.hpp"

namespace geantx {
class ProxyMollerScattering : public ProxyEmModel<ProxyMollerScattering> {

public:
  GEANT_HOST
  ProxyMollerScattering() { fLowEnergyLimit = 100.0 * clhep::eV; }

  GEANT_HOST_DEVICE
  ProxyMollerScattering(int tid) : ProxyEmModel(tid)
  { fLowEnergyLimit = 100.0 * clhep::eV; }

  GEANT_HOST
  ProxyMollerScattering(const ProxyMollerScattering &model) 
   : ProxyEmModel<ProxyMollerScattering>() { this->fRng = model.fRng; }

  GEANT_HOST_DEVICE
  ~ProxyMollerScattering() {};

  //mandatory methods
  GEANT_HOST_DEVICE
  double CrossSectionPerAtom(double Z, double kineticEnergy);

  GEANT_HOST_DEVICE
  int SampleSecondaries(TrackState *track);

  //auxiliary methods

private:

  friend class ProxyEmModel<ProxyMollerScattering>;

};

// based on Geant4 processes/electromagnetic/standard/src/G4MollerBhabhaModel
GEANT_HOST_DEVICE
double ProxyMollerScattering::CrossSectionPerAtom(double Z, double kineticEnergy)
{
  double xsec = 0.0;

  //TODO: MaterialCuts by an argument
  const double cutEnergy = 10.0 * clhep::eV; // matcut->GetProductionCutsInEnergy()[1];

  double tmax = vecCore::math::Min(fHighEnergyLimit, 0.5*kineticEnergy);

  if(cutEnergy > tmax) return xsec;

  double xmin  = cutEnergy/kineticEnergy;
  double xmax  = tmax/kineticEnergy;
  double tau   = kineticEnergy/clhep::electron_mass_c2;
  double gam   = tau + 1.0;
  double gamma2= gam*gam;
  double beta2 = tau*(tau + 2)/gamma2;
  
  //Moller (e-e-) scattering
  double gg = (2.0*gam - 1.0)/gamma2;
  xsec = ((xmax - xmin)*(1.0 - gg + 1.0/(xmin*xmax)
			  + 1.0/((1.0-xmin)*(1.0 - xmax)))
	  - gg*vecCore::math::Log( xmax*(1.0 - xmin)/(xmin*(1.0 - xmax)) ) ) / beta2;
  
  xsec *= clhep::twopi_mc2_rcl2/kineticEnergy;
  
  return Z * xsec;
}

// based on Geant4 processes/electromagnetic/standard/src/G4MollerBhabhaModel
GEANT_HOST_DEVICE
int ProxyMollerScattering::SampleSecondaries(TrackState *track)
{
  int nsecondaries = 0;
  double kineticEnergy = track->fPhysicsState.fEkin;

  //cut energy
  double maxEnergy = 1.0 * clhep::TeV; //temp
  double cutEnergy = 1.0 * clhep::keV; //temp

  double tmin = cutEnergy;  
  double tmax = 0.5*kineticEnergy; 

  if(maxEnergy < tmax) { tmax = maxEnergy; }
  if(tmin >= tmax) { return nsecondaries; }

  double energy = kineticEnergy + clhep::electron_mass_c2;
  double xmin   = tmin/kineticEnergy;
  double xmax   = tmax/kineticEnergy;
  double gam    = energy/clhep::electron_mass_c2;
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
  double totalMomentum = vecCore::math::Sqrt(kineticEnergy  * (kineticEnergy + 2.0* clhep::electron_mass_c2));
  double deltaMomentum = vecCore::math::Sqrt(deltaKinEnergy * (deltaKinEnergy + 2.0* clhep::electron_mass_c2));
  double cost =  deltaKinEnergy * (energy +  clhep::electron_mass_c2) / (deltaMomentum * totalMomentum);
  if(cost > 1.0) { cost = 1.0; }
  double sint = vecCore::math::Sqrt((1.0 - cost)*(1. + cost));

  double phi = clhep::twopi * this->fRng->uniform();

  double xhat = sint*vecCore::math::Cos(phi);
  double yhat = sint*vecCore::math::Sin(phi);
  double zhat = cost;
  
  Math::RotateToLabFrame(xhat, yhat, zhat, track->fDir.x(), track->fDir.y(), track->fDir.z());
  ThreeVector deltaDirection(xhat, yhat, zhat);
  
  //TODO: create a delta ray (electron) and push it to the secondary container 
  TrackState electron;
  electron.fDir = deltaDirection;
  electron.fPhysicsState.fEkin = deltaKinEnergy;

  ++nsecondaries;

  //update the primary
  
  track->fPhysicsState.fEkin -= deltaKinEnergy;
  track->fDir = (totalMomentum*track->fDir - deltaMomentum*deltaDirection).Unit();

  return nsecondaries;
}

} // namespace geantx
