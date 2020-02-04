//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/cuts/src/CutConverterForGamma.cpp
 * @brief Originated in the GeantV project
 */
//===----------------------------------------------------------------------===//

#include "Geant/cuts/CutConverterForGamma.hpp"

// from material
#include "Geant/material/Types.hpp"
#include "Geant/material/Material.hpp"
#include "Geant/material/MaterialProperties.hpp"
#include "Geant/material/Element.hpp"

#include "Geant/core/math_wrappers.hpp"

#include <cmath>
#include <iostream>

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

CutConverterForGamma::CutConverterForGamma(int numebins, double mincutenergy,
                                           double maxcutenergy)
    : CutConverter(0, numebins, mincutenergy, maxcutenergy), fZ(-1.), fS200keV(0.),
      fTmin(0.), fSmin(0.), fCmin(0.), fTlow(0.), fSlow(0.), fS1keV(0.), fClow(0.),
      fChigh(0.)
{
  if (fMinCutEnergy >= fMaxCutEnergy) {
    std::cerr << "  *** ERROR in CutConverterForGamma::CutConverterForGamma() "
              << std::endl
              << "       minenergy = " << mincutenergy / geantx::units::GeV
              << " [GeV] >= maxenergy = " << maxcutenergy / geantx::units::GeV << " [GeV]"
              << std::endl;
    exit(-1);
  }
  Initialise();
}

// must be called before using the Convert method if new element has been inserted into
// the Element table!
void CutConverterForGamma::Initialise()
{
  CutConverter::Initialise();
  BuildElossOrAbsXsecTable();
}

CutConverterForGamma::~CutConverterForGamma() {}

void CutConverterForGamma::BuildLengthVector(const Material *mat)
{
  const Vector_t<Element *> elemVect = mat->GetElementVector();
  const double *theAtomicNumDensityVector =
      mat->GetMaterialProperties()->GetNumOfAtomsPerVolumeVect();
  int numElements = elemVect.size();

  double maxAbsLenght = -1.0;
  for (int iener = 0; iener < fNumEBins; ++iener) {
    double macXsec = 0.0;
    for (int ielem = 0; ielem < numElements; ++ielem) {
      int izet     = std::lrint(elemVect[ielem]->GetZ());
      double *vect = fElossOrAbsXsecTable[izet - 1];
      macXsec += theAtomicNumDensityVector[ielem] * vect[iener];
    }
    double absorptionLength = 5.0 / macXsec;
    fLengthVector[iener]    = absorptionLength;
    if (maxAbsLenght < absorptionLength) {
      maxAbsLenght   = absorptionLength;
      fMaxLengthIndx = iener;
    }
  }
}

// Compute the photon "absorption" cross section: sum of destructive (approximated) cross
// sections like pair production, Compton scattering and photoelectric effect (taken from
// Geant4)
double CutConverterForGamma::ComputeELossOrAbsXsecPerAtom(double zet, double ekin)
{
  const double t1keV   = 1.0 * geantx::units::keV;
  const double t200keV = 200.0 * geantx::units::keV;
  const double t100MeV = 100.0 * geantx::units::MeV;
  //  compute Z dependent quantities if the cached Z is different than zet
  if (std::abs(zet - fZ) > 0.1) {
    fZ                = zet;
    double Zsquare    = fZ * fZ;
    double Zlog       = Math::Log(fZ);
    double Zlogsquare = Zlog * Zlog;
    // set some Z dependent variables
    fS200keV = (0.2651 - 0.1501 * Zlog + 0.02283 * Zlogsquare) * Zsquare;
    fTmin    = (0.552 + 218.5 / fZ + 557.17 / Zsquare) * geantx::units::MeV;
    fSmin = (0.01239 + 0.005585 * Zlog - 0.000923 * Zlogsquare) * Math::Exp(1.5 * Zlog);
    fCmin = Math::Log(fS200keV / fSmin) /
            (Math::Log(fTmin / t200keV) * Math::Log(fTmin / t200keV));
    fTlow  = 0.2 * Math::Exp(-7.355 / std::sqrt(fZ)) * geantx::units::MeV;
    fSlow  = fS200keV * Math::Exp(0.042 * fZ * Math::Log(t200keV / fTlow) *
                                 Math::Log(t200keV / fTlow));
    fS1keV = 300.0 * Zsquare;
    fClow  = Math::Log(fS1keV / fSlow) / Math::Log(fTlow / t1keV);
    fChigh = (7.55e-5 - 0.0542e-5 * fZ) * Zsquare * fZ / Math::Log(t100MeV / fTmin);
  }
  // calculate the absorption cross section (using an approximate empirical formula)
  double xs = 0.0;
  if (ekin < fTlow) {
    if (ekin < t1keV)
      xs = fSlow * Math::Exp(fClow * Math::Log(fTlow / t1keV));
    else
      xs = fSlow * Math::Exp(fClow * Math::Log(fTlow / ekin));
  } else if (ekin < t200keV) {
    xs = fS200keV *
         Math::Exp(0.042 * fZ * Math::Log(t200keV / ekin) * Math::Log(t200keV / ekin));
  } else if (ekin < fTmin) {
    double dum = Math::Log(fTmin / ekin);
    xs         = fSmin * Math::Exp(fCmin * dum * dum);
  } else {
    xs = fSmin + fChigh * Math::Log(ekin / fTmin);
  }
  return xs * geantx::units::barn;
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
