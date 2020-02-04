//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/cuts/CutConverterForGamma.hpp
 * @brief Originated in the GeantV project
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/cuts/CutConverter.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

class Material;
/**
 * @brief   Production threshold converter for gamma.
 * @class   CutConverterForGamma
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class CutConverterForGamma : public CutConverter {
public:
  CutConverterForGamma(int numebins        = 301,
                       double mincutenergy = 100.0 * geantx::units::eV,
                       double maxcutenergy = 10.0 * geantx::units::GeV);
  virtual ~CutConverterForGamma();

  virtual void Initialise();

protected:
  virtual void BuildLengthVector(const Material *mat);
  virtual double ComputeELossOrAbsXsecPerAtom(double zet, double ekin);

private:
  // some Z dependent cached variables for the approximated absorption cross section
  // computation
  double fZ;
  double fS200keV;
  double fTmin;
  double fSmin;
  double fCmin;
  double fTlow;
  double fSlow;
  double fS1keV;
  double fClow;
  double fChigh;
};

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
