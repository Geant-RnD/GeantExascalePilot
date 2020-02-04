//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/cuts/CutConverterForElectron.hpp
 * @brief Originated in the GeantV project
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/cuts/CutConverter.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

/**
 * @brief   Production threshold converter for electron.
 * @class   CutConverterForElectron
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class CutConverterForElectron : public CutConverter {
public:
  CutConverterForElectron(int numebins        = 301,
                          double mincutenergy = 100.0 * geantx::units::eV,
                          double maxcutenergy = 10.0 * geantx::units::GeV);
  virtual ~CutConverterForElectron();

  virtual void Initialise();

protected:
  virtual double ComputeELossOrAbsXsecPerAtom(double zet, double ekin);
};

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
