
#include "Geant/cuts/CutConverterForElectron.hpp"

#include "Geant/material/Types.hpp"
#include "Geant/material/Material.hpp"
#include "Geant/material/MaterialProperties.hpp"
#include "Geant/material/Element.hpp"

#include "Geant/core/math_wrappers.hpp"

#include <cmath>
#include <iostream>

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

CutConverterForElectron::CutConverterForElectron(int numebins, double mincutenergy, double maxcutenergy)
    : CutConverter(1, numebins, mincutenergy, maxcutenergy)
{
  if (fMinCutEnergy >= fMaxCutEnergy) {
    std::cerr << "  *** ERROR in CutConverterForElectron::CutConverterForElectron() " << std::endl
              << "       minenergy = " << mincutenergy / geantx::units::GeV
              << " [GeV] >= maxenergy = " << maxcutenergy / geantx::units::GeV << " [GeV]" << std::endl;
    exit(-1);
  }
  Initialise();
}

// must be called before using the Convert method if new element has been inserted into the Element table!
void CutConverterForElectron::Initialise()
{
  CutConverter::Initialise();
  BuildElossOrAbsXsecTable();
}

CutConverterForElectron::~CutConverterForElectron()
{
}

double CutConverterForElectron::ComputeELossOrAbsXsecPerAtom(double zet, double ekin)
{
  const double cbr1  = 0.02;
  const double cbr2  = -5.7e-5;
  const double cbr3  = 1.;
  const double cbr4  = 0.072;
  const double tlow  = 10. * geantx::units::keV;
  const double thigh = 1. * geantx::units::GeV;
  const double mass  = geantx::units::kElectronMassC2;
  const double taul  = tlow / mass;
  const double cpot  = 1.6e-5 * geantx::units::MeV;
  const double fact  = geantx::units::kTwoPi * geantx::units::kElectronMassC2 * geantx::units::kClassicElectronRadius *
                      geantx::units::kClassicElectronRadius;

  double ionpot    = cpot * Math::Exp(0.9 * Math::Log(zet)) / mass;
  double ionpotlog = Math::Log(ionpot);

  // calculate approximated dE/dx for electrons
  double tau  = ekin / mass;
  double dEdx = 0.0;
  if (tau < taul) {
    double t1    = taul + 1.;
    double t2    = taul + 2.;
    double tsq   = taul * taul;
    double beta2 = taul * t2 / (t1 * t1);
    double f    = 1. - beta2 + Math::Log(tsq / 2.) + (0.5 + 0.25 * tsq + (1. + 2. * taul) * Math::Log(0.5)) / (t1 * t1);
    dEdx        = (Math::Log(2. * taul + 4.) - 2. * ionpotlog + f) / beta2;
    dEdx        = fact * zet * dEdx;
    double clow = dEdx * std::sqrt(taul);
    dEdx        = clow / std::sqrt(tau);
  } else {
    double t1    = tau + 1.;
    double t2    = tau + 2.;
    double tsq   = tau * tau;
    double beta2 = tau * t2 / (t1 * t1);
    double f     = 1. - beta2 + Math::Log(tsq / 2.) + (0.5 + 0.25 * tsq + (1. + 2. * tau) * Math::Log(0.5)) / (t1 * t1);
    dEdx         = (Math::Log(2. * tau + 4.) - 2. * ionpotlog + f) / beta2;
    dEdx         = fact * zet * dEdx;
    // loss from bremsstrahlung follows
    double cbrem = (cbr1 + cbr2 * zet) * (cbr3 + cbr4 * Math::Log(ekin / thigh));
    cbrem        = 0.1 * zet * (zet + 1.) * cbrem * tau / beta2;
    dEdx += fact * cbrem;
  }
  return dEdx;
}

} // namespace GEANT_IMPL_NAMESPACE                                                                                                                           
} // namespace geantx
