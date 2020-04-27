//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/src/ProxyEgammaGun.cpp
 * @brief ProxyEgammaGun
 */
//===----------------------------------------------------------------------===//

#include "Geant/proxy/ProxyEgammaGun.hpp"
#include "Geant/core/math_wrappers.hpp"

namespace geantx {

ProxyEgammaGun::ProxyEgammaGun(int nparticles) 
  : ProxyEventGenerator<ProxyEgammaGun>(), 
    fNumberOfParticles(nparticles), fGammaFraction(0.5) 
{ 
  fRng = new ProxyRandom;
}

ProxyEgammaGun::ProxyEgammaGun(int nevents, int nparticles)
  : ProxyEventGenerator<ProxyEgammaGun>(nevents), 
    fNumberOfParticles(nparticles), fGammaFraction(0.5)
{ 
  fRng = new ProxyRandom;
}

void ProxyEgammaGun::SetDirectionXYZ(double x, double y, double z)
{
  double rho = Math::Sqrt(x*x+y*y);
  double theta = std::atan(rho/z);
 
  fMinEta = fMaxEta = -Math::Log(std::tan(0.5*theta));
  fMinPhi = fMaxPhi = std::atan(y/z);
}

ProxyEvent* ProxyEgammaGun::GenerateOneEvent()
{ 
  ProxyVertex *primaryVertex = new ProxyVertex();

  for( int i = 0 ; i < fNumberOfParticles ; ++i) {
    TrackState* atrack = new TrackState();

    int pid = -1;
    double charge = 0;

    if ( fRng->uniform() < fGammaFraction ) {
      pid = 22;
      charge = 0.0;
    }
    else {
      pid = 11;
      charge = -1.0;
    }

    atrack->fSchedulingState.fGVcode =  pid;

    double p = fMinP + (fMaxP - fMinP)*fRng->uniform();
    double mass = clhep::electron_mass_c2*charge*charge;

    atrack->fPhysicsState.fMomentum = p;
    atrack->fPhysicsState.fEkin = p*p/(sqrt(p*p + mass*mass) + mass);

    double eta = fMinEta + (fMaxEta - fMinEta)*fRng->uniform();
    double phi = fMinPhi + (fMaxPhi - fMinPhi)*fRng->uniform();
    double theta = 2.*std::atan(std::exp(-eta));
    atrack->fDir = {std::sin(theta)*std::cos(phi),
		    std::sin(theta)*std::sin(phi), std::cos(theta)};

    //TODO: add other quantities

    primaryVertex->AddParticle(atrack);
  }

  ProxyEvent *anEvent = new ProxyEvent();
  anEvent->AddVertex(primaryVertex);

  return anEvent;
}

} // namespace geantx
