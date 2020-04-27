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
 * @file Geant/proxy/ProxyEgammaGun.hpp
 * @brief a generator for random particles 
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/proxy/ProxyEvent.hpp"
#include "Geant/proxy/ProxyEventGenerator.hpp"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"
#include "Geant/proxy/ProxyRandom.hpp"

namespace geantx
{

class ProxyEgammaGun;

template <> struct Generator_traits<ProxyEgammaGun>
{
  using Generator_t = ProxyEgammaGun;
};

class ProxyEgammaGun : public ProxyEventGenerator<ProxyEgammaGun>
{
  using this_type = ProxyEgammaGun;
  friend class ProxyEventGenerator<ProxyEgammaGun>;

public:
  
  ProxyEgammaGun(int nparticles = 1); 
  ProxyEgammaGun(int nevents, int nparticles = 1);

  ~ProxyEgammaGun() = default;
  
  // mandatory method
  ProxyEvent* GenerateOneEvent(); 

  // auxiliary method
  void SetNumberOfParticles(int np) { fNumberOfParticles = np; };

  void SetGammaFraction(double frac) { fGammaFraction = frac; };

  void SetMomentum(double min, double max) { fMinP = min, fMaxP = max; }

  void SetEta(double min, double max) { fMinEta = min, fMaxEta = max; }

  void SetPhi(double min, double max) { fMinPhi = min, fMaxPhi = max; }

  void SetDirectionXYZ(double x, double y, double z);

private:
  int fNumberOfParticles;
  double fGammaFraction = 0.5;
  double fMinP = 1.0*clhep::GeV;
  double fMaxP = 1.0*clhep::TeV;
  double fMinEta = -3;
  double fMaxEta =  3;
  double fMinPhi = 0;
  double fMaxPhi = clhep::twopi;

  ProxyRandom *fRng = nullptr;
};

}  // namespace geantx
