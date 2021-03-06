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
 * @file Geant/proxy/ProxySeltzerBerger.hpp 
 * @brief the SeltzerBerger model of the electron Bremsstrahlung
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/proxy/ProxySystemOfUnits.hpp"
#include "Geant/proxy/ProxyPhysicalConstants.hpp"
#include "Geant/core/math_wrappers.hpp"

#include "Geant/proxy/ProxyEmModel.hpp"
#include "Geant/proxy/Proxy2DVector.hpp"
#include "Geant/proxy/ProxyPhysicsTable.cuh"

#include <VecCore/VecCore>

namespace geantx {
class ProxySeltzerBerger : public ProxyEmModel<ProxySeltzerBerger> {

public:

  GEANT_HOST
  ProxySeltzerBerger();

  GEANT_HOST_DEVICE
  ProxySeltzerBerger(int tid)
    : ProxyEmModel(tid) { fLowEnergyLimit = 100.0 * clhep::eV; }

 
  GEANT_HOST_DEVICE
  ProxySeltzerBerger(const ProxySeltzerBerger &model) 
    : ProxyEmModel<ProxySeltzerBerger>() { this->fRng = model.fRng; }

  GEANT_HOST_DEVICE
  ~ProxySeltzerBerger();

  //mandatory methods
  GEANT_HOST
  void Initialization();

  GEANT_HOST_DEVICE
  double CrossSectionPerAtom(double Z, double kineticEnergy);

  GEANT_HOST_DEVICE
  int SampleSecondaries(TrackState *track);

private:

  //auxiliary methods
  GEANT_HOST 
  bool RetrieveSeltzerBergerData(std::ifstream &in, Proxy2DVector *vec2D);

  GEANT_HOST_DEVICE
  ThreeVector SamplePhotonDirection(double energyIn);

  GEANT_HOST_DEVICE
  double ComputeRelDXSectionPerAtom(double gammaEnergy);

  GEANT_HOST_DEVICE
  double ComputeDXSectionPerAtom(double gammaEnergy);

  GEANT_HOST_DEVICE
  void CalcLPMFunctions(double k, double totalEnergy);

  GEANT_HOST_DEVICE
  double Phi1(double gg);

  GEANT_HOST_DEVICE
  double Phi1M2(double gg);

  GEANT_HOST_DEVICE
  double Psi1(double eps);

  GEANT_HOST_DEVICE
  double Psi1M2(double eps);

  friend class ProxyEmModel<ProxySeltzerBerger>;

private:

  double currentZ;
  double densityFactor;

  double lpmEnergy;
  double xiLPM;
  double phiLPM;
  double gLPM;

  double z13;
  double z23;
  double lnZ;
  double Fel;
  double Finel;
  double fCoulomb;
  double fMax;

  Proxy2DVector *fDataSB;
  //  ProxyPhysicsTable *fLambdaTable;
};

GEANT_HOST
ProxySeltzerBerger::ProxySeltzerBerger() 
{ 
  fLowEnergyLimit = 100.0 * clhep::eV; 

  densityFactor = 0;
  currentZ = 0;

  lpmEnergy = xiLPM = phiLPM = gLPM = z13 = z23 = lnZ = Fel =
  Finel = fCoulomb = fMax = 0;
  // TODO - comment out when completed
  Initialization();
}

GEANT_HOST
ProxySeltzerBerger::~ProxySeltzerBerger() 
{ 
  free(fDataSB);
}

GEANT_HOST
void ProxySeltzerBerger::Initialization()
{
  fDataSB = (Proxy2DVector *)malloc(data::maximumZ * sizeof(Proxy2DVector));

  char sbDataFile[256];

  for (int iZ = 0; iZ < data::maximumZ; iZ++) {
    sprintf(sbDataFile, "data/brem_SB/br%d", iZ + 1);
    std::ifstream fin(sbDataFile);
    bool check = RetrieveSeltzerBergerData(fin, &fDataSB[iZ]);
    if (!check) {
      printf("Failed To open SeltzerBerger Data file for Z= %d\n", iZ + 1);
    }
  }

  //  fLambdaTable = new ProxyPhysicsTable();  

}

GEANT_HOST_DEVICE
double ProxySeltzerBerger::CrossSectionPerAtom(double Z, double kineticEnergy)
{
  double xsec = 0.0;

  //get cut from the material cut
  double cut = 10.0*clhep::keV;

  // number of intervals and integration step
  double totalEnergy = kineticEnergy + clhep::electron_mass_c2;
  double vcut        = vecCore::math::Log(cut / totalEnergy);
  double vmax        = vecCore::math::Log(kineticEnergy / totalEnergy);
  int n              = vecCore::math::Floor(0.45 * (vmax - vcut)) + 4;
  double delta = (vmax - vcut)/n;

  double e0 = vcut;
  double xs = 0.0;

  //get densityFactor from the material cut
  // densityFactor = mat->GetElectronDensity()*fMigdalConstant;
  // energyThresholdLPM=math::Sqrt(densityFactor)*lpmEnergy;
  densityFactor      = 1.0;
  double energyThresholdLPM = 1.0;
  double densityCorr        = densityFactor * totalEnergy * totalEnergy;

  //  double xgi[8] = {0.0199, 0.1017, 0.2372, 0.4083, 0.5917, 0.7628, 0.8983, 0.9801};
  //  double wgi[8] = {0.0506, 0.1112, 0.1569, 0.1813, 0.1813, 0.1569, 0.1112, 0.0506};

  // integration
  for (int l = 0; l < 8 ; l++) {
    for (int i = 0; i < 8; i++) {

      double eg = vecCore::math::Exp(e0 + data::xgi[i] * delta) * totalEnergy;

      if (totalEnergy > energyThresholdLPM) {
	xs = ComputeRelDXSectionPerAtom(eg);
      } else {
        // TODo:syjun - select G4eBremsstrahlungRelMedel or G4SeltzerBergerModel
	xs = ComputeDXSectionPerAtom(eg);
      }
      xsec += data::wgi[i] * xs / (1.0 + densityCorr / (eg * eg));
    }
    e0 += delta;
  }

  xsec *= delta;

  return xsec;
}

GEANT_HOST_DEVICE
int ProxySeltzerBerger::SampleSecondaries(TrackState *track)
{
  int nsecondaries = 0;
  double kineticEnergy = track->fPhysicsState.fEkin;

  // G4SeltzerBergerModel::SampleSecondaries
  //  double cut  = Min(cutEnergy, kineticEnergy);
  //  double emax = Min(maxEnergy, kineticEnergy);
  //@@@syj cutEnergy should be get from the material table (cut table).
  // other hard coded numbers are also temporary and will be replaced properly
  double cut  = vecCore::math::Min(1.0 * clhep::keV, kineticEnergy);
  double emax = vecCore::math::Min(1.0 * clhep::TeV, kineticEnergy);
  if (cut >= emax) {
    return nsecondaries;
  }

  //TODO: select random Z 
  int Z = 10; 

  double totalEnergy = kineticEnergy + clhep::electron_mass_c2;
  double densityCorr = densityFactor * totalEnergy * totalEnergy;
  double xmin = vecCore::math::Log(cut * cut + densityCorr);
  double xmax = vecCore::math::Log(emax * emax + densityCorr);
  double y    = vecCore::math::Log(kineticEnergy / clhep::MeV);

  double v;

  // majoranta
  double x0   = cut / kineticEnergy;
  double vmax = fDataSB[Z].Value(x0, y) * 1.02;

  double_t epeaklimit = 300 * clhep::MeV;
  double_t elowlimit  = 10 * clhep::keV;

  // majoranta corrected for e-
  bool isElectron = true;
  if (isElectron && x0 < 0.97 && ((kineticEnergy > epeaklimit) || (kineticEnergy < elowlimit))) {
    double ylim = vecCore::math::Min(fDataSB[Z].Value(0.97, 4 * vecCore::math::Log(10.)), 1.1 * fDataSB[Z].Value(0.97, y));
    if (ylim > vmax) {
      vmax = ylim;
    }
  }
  if (x0 < 0.05) {
    vmax *= 1.2;
  }

  double gammaEnergy = 0;

  do {
    double auxrand = this->fRng->uniform();
    double x       = vecCore::math::Exp(xmin + auxrand * (xmax - xmin)) - densityCorr;
    if (x < 0.0) {
      x = 0.0;
    }
    gammaEnergy = vecCore::math::Sqrt(x);
    double x1   = gammaEnergy / kineticEnergy;
    v           = fDataSB[Z].Value(x1, y);

    // correction for positrons

  } while (v < vmax * this->fRng->uniform());

  //TODO: create a bremsstrahlung photon and push it to the secondary container
  TrackState photon;

  photon.fPhysicsState.fEkin = gammaEnergy;

  ThreeVector gammaDirection = SamplePhotonDirection(gammaEnergy);
  Math::RotateToLabFrame(gammaDirection.x(), gammaDirection.y(), gammaDirection.z(), 
			 track->fDir.x(), track->fDir.y(), track->fDir.z());

  photon.fDir = gammaDirection;

  ++nsecondaries;

  //update the primary
  double totalMomentum = vecCore::math::Sqrt(kineticEnergy*(totalEnergy + clhep::electron_mass_c2));
  track->fPhysicsState.fEkin -= gammaEnergy;
  track->fDir = (totalMomentum*track->fDir - gammaEnergy*gammaDirection).Unit();

  return nsecondaries;
}

GEANT_HOST_DEVICE
ThreeVector ProxySeltzerBerger::SamplePhotonDirection(double energyIn)
{
  // angle of the radiated photon
  // based on G4DipBustGenerator::SampleDirection
  
  double c     = 4.0 - 8.0 * this->fRng->uniform();
  double signc = vecCore::math::Sign(c);
  double a     = vecCore::math::Abs(c);

  double delta = 0.5 * (vecCore::math::Sqrt(4.0 + a * a) + a);

  double cofA = -signc * vecCore::math::Pow(delta, 1.0/ 3.0);

  double cosTheta = cofA - 1.0 / cofA;

  double tau  = energyIn / clhep::electron_mass_c2;
  double beta = vecCore::math::Sqrt(tau * (tau + 2.0)) / (tau + 1.0);

  cosTheta = (cosTheta + beta) / (1.0 + cosTheta * beta);

  double sinTheta = vecCore::math::Sqrt((1.0 - cosTheta) * (1.0 + cosTheta));
  double phi = clhep::twopi * this->fRng->uniform();

  ThreeVector gammaDirection(sinTheta*vecCore::math::Cos(phi),sinTheta*vecCore::math::Sin(phi),cosTheta);

  return gammaDirection;
}

GEANT_HOST_DEVICE
double ProxySeltzerBerger::ComputeRelDXSectionPerAtom(double gammaEnergy)
{
// Ultra relativistic model
//   only valid for very high energies, but includes LPM suppression
//    * complete screening

  if (gammaEnergy < 0.0) return 0.0;

  double totalEnergy = gammaEnergy +  clhep::electron_mass_c2;

  double y     = gammaEnergy / totalEnergy;
  double y2    = y * y * .25;
  double yone2 = (1. - y + 2. * y2);

  // ** form factors complete screening case **

  // ** calc LPM functions: include ter-mikaelian merging with density effect **
  //  double xiLPM, gLPM, phiLPM;  // to be made member variables !!!

  CalcLPMFunctions(gammaEnergy,totalEnergy);

  double mainLPM    = xiLPM * (y2 * gLPM + yone2 * phiLPM) * ((Fel - fCoulomb) + Finel / currentZ);
  double secondTerm = (1. - y) / 12. * (1. + 1. / currentZ);

  double cross = mainLPM + secondTerm;
  return cross;
}

GEANT_HOST_DEVICE
double ProxySeltzerBerger::ComputeDXSectionPerAtom(double gammaEnergy)
{
// Relativistic model
//  only valid for high energies (and if LPM suppression does not play a role)
//  * screening according to thomas-fermi-Model (only valid for Z > 5)
//  * no LPM effect

  if (gammaEnergy < 0.0) return 0.0;

  double totalEnergy = gammaEnergy + clhep::electron_mass_c2;
  double y    = gammaEnergy / totalEnergy;

  double main = 0., secondTerm = 0.;

  //@@@ use_completescreening(false) in G4eBremsstrahlungRelModel constructor
  // if (use_completescreening|| currentZ<5) {
  if (currentZ < 5) {
    // ** form factors complete screening case **
    main       = (3. / 4. * y * y - y + 1.) * ((Fel - fCoulomb) + Finel / currentZ);
    secondTerm = (1. - y) / 12. * (1. + 1. / currentZ);
  } else {
    // ** intermediate screening using Thomas-Fermi FF from Tsai
    // only valid for Z >= 5 **
    double dd  = 100. * clhep::electron_mass_c2 * y / (totalEnergy - gammaEnergy);
    double gg  = dd / z13;
    double eps = dd / z23;
    double phi1   = Phi1(gg);
    double phi1m2 = Phi1M2(gg);
    double psi1   = Psi1(eps);
    double psi1m2 = Psi1M2(eps);
    main = (3. / 4. * y * y - y + 1.) *
	((0.25 * phi1 - 1. / 3. * lnZ - fCoulomb) + (0.25 * psi1 - 2. / 3. * lnZ) / currentZ);
    secondTerm = (1. - y) / 8. * (phi1m2 + psi1m2 / currentZ);
  }
  double cross = main + secondTerm;
  return cross;
}

GEANT_HOST_DEVICE
void ProxySeltzerBerger::CalcLPMFunctions(double k, double totalEnergy)
{
  // *** calculate lpm variable s & sprime ***
  // Klein eqs. (78) & (79)

  double sprime = vecCore::math::Sqrt(0.125 * k * lpmEnergy / (totalEnergy * (totalEnergy - k)));
  double s1 = (1. / (184.15 * 184.15)) * z23;
  double logS1 = 2. / 3. * lnZ - 2. * vecCore::math::Log(184.15);
  double logTS1 = vecCore::math::Log(2.) + logS1;

  xiLPM = 2.;

  if (sprime > 1)
    xiLPM = 1.;
  else if (sprime > vecCore::math::Sqrt(2.) * s1) {
    double h = vecCore::math::Log(sprime) / logTS1;
     xiLPM = 1 + h - 0.08 * (1 - h) * (1 - (1 - h) * (1 - h)) / logTS1;
  }

  double s0 = sprime / vecCore::math::Sqrt(xiLPM);

  // *** merging with density effect***  should be only necessary in region
  // "close to" kp, e.g. k<100*kp using Ter-Mikaelian eq. (20.9)
  double k2 = k * k;
  double densityCorr = densityFactor*totalEnergy*totalEnergy;
  s0 *= (1 + (densityCorr / k2));

  // recalculate Xi using modified s above
  // Klein eq. (75)
  xiLPM = 1.;
  if (s0 <= s1)
    xiLPM = 2.;
  else if ((s1 < s0) && (s0 <= 1))
    xiLPM = 1. + vecCore::math::Log(s0) / logS1;

  // *** calculate supression functions phi and G ***
  // Klein eqs. (77)
  double s2 = s0 * s0;
  double s3 = s0 * s2;
  double s4 = s2 * s2;
  if (s0 < 0.1) {
    // high suppression limit
    phiLPM = 6. * s0 - 18.84955592153876 * s2 + 39.47841760435743 * s3 - 57.69873135166053 * s4;
    gLPM   = 37.69911184307752 * s2 - 236.8705056261446 * s3 + 807.7822389 * s4;
  } else if (s0 < 1.9516) {
    // intermediate suppression
    // using eq.77 approxim. valid s<2.
    phiLPM = 1. - vecCore::math::Exp(-6. * s0 * (1. + (3. - Math::Pi()) * s0) + s3 / (0.623 + 0.795 * s0 + 0.658 * s2));
    if (s0 < 0.415827397755) {
  	// using eq.77 approxim. valid 0.07<s<2
  	double psiLPM = 1 - vecCore::math::Exp(-4 * s0 - 8 * s2 / (1 + 3.936 * s0 + 4.97 * s2 - 0.05 * s3 + 7.50 * s4));
  	gLPM          = 3 * psiLPM - 2 * phiLPM;
    } else {
  	// using alternative parametrisiation
    double pre = -0.16072300849123999 + s0 * 3.7550300067531581 + s2 * -1.7981383069010097 +
  	s3 * 0.67282686077812381 + s4 * -0.1207722909879257;
    gLPM = vecCore::math::Tanh(pre);
    }
  } else {
    // low suppression limit valid s>2.
    phiLPM = 1. - 0.0119048 / s4;
    gLPM   = 1. - 0.0230655 / s4;
  }

  // *** make sure suppression is smaller than 1 ***
  // *** caused by Migdal approximation in xi    ***
  if (xiLPM * phiLPM > 1. || s0 > 0.57) {
    xiLPM = 1. / phiLPM;
  }
}

GEANT_HOST_DEVICE
double ProxySeltzerBerger::Phi1(double gg)
{
  // Thomas-Fermi FF from Tsai, eq.(3.38) for Z>=5
  return 20.863 - 2. * vecCore::math::Log(1. + (0.55846 * gg) * (0.55846 * gg)) -
    4. * (1. - 0.6 * vecCore::math::Exp(-0.9 * gg) - 0.4 * vecCore::math::Exp(-1.5 * gg));
}

GEANT_HOST_DEVICE
double ProxySeltzerBerger::Phi1M2(double gg)
{
  // Thomas-Fermi FF from Tsai, eq. (3.39) for Z>=5
  return 2. / (3. * (1. + 6.5 * gg + 6. * gg * gg));
}

GEANT_HOST_DEVICE
double ProxySeltzerBerger::Psi1(double eps)
{
  // Thomas-Fermi FF from Tsai, eq.(3.40) for Z>=5
  return 28.340 - 2. * vecCore::math::Log(1. + (3.621 * eps) * (3.621 * eps)) -
    4. * (1. - 0.7 * vecCore::math::Exp(-8 * eps) - 0.3 * vecCore::math::Exp(-29. * eps));
}

GEANT_HOST_DEVICE
double ProxySeltzerBerger::Psi1M2(double eps)
{
  // Thomas-Fermi FF from Tsai, eq. (3.41) for Z>=5
  return 2. / (3. * (1. + 40. * eps + 400. * eps * eps));
}

GEANT_HOST 
bool ProxySeltzerBerger::RetrieveSeltzerBergerData(std::ifstream &in, Proxy2DVector *vec2D)
{
  // binning
  int k;
  int dummyX; // 32 fixed up to Z = 92
  int dummyY; // 57 fixed up to Z = 92
  in >> k >> dummyX >> dummyY;
  if (in.fail()) {
    return false;
  }

  // contents
  double valx, valy, val;
  for (size_t i = 0; i < data::numberOfXNodes; ++i) {
    in >> valx;
    if (in.fail()) {
	return false;
    }
    vec2D->PutX(i, valx);
  }
  for (size_t j = 0; j < data::numberOfYNodes; ++j) {
    in >> valy;
    if (in.fail()) {
	return false;
    }
    vec2D->PutY(j, valy);
  }
  for (size_t j = 0; j < data::numberOfYNodes; ++j) {
    for (size_t i = 0; i < data::numberOfXNodes; ++i) {
	in >> val;
	if (in.fail()) {
	  return false;
	}
	vec2D->PutValue(i, j, val);
    }
  }
  in.close();
  return true;
}

} // namespace geantx
