//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/cuts/CutConverter.hpp
 * @brief Originated in the GeantV project
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/SystemOfUnits.hpp"
#include "Geant/material/Material.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

class Material;

// class Material;

/**
 * @brief   Base class to convert production threshold given in length to energy or given in energy to length.
 * @class   CutConverter
 * @author  M Novak, A Ribon
 * @date    april 2016
 */
class CutConverter {
public:
  CutConverter(int particleindx = -1, int numebins = 301, double mincutenergy = 100.0 * geantx::units::eV,
               double maxcutenergy = 10.0 * geantx::units::GeV);
  virtual ~CutConverter();

  virtual double Convert(const Material *mat, double cut, bool isfromlength = true);
  virtual void Initialise();

  void SetProductionCutRangeInEnergy(double minenergycut, double maxenergycut);

protected:
  virtual void BuildLengthVector(const Material *mat);
  virtual void BuildElossOrAbsXsecTable();
  virtual double ComputeELossOrAbsXsecPerAtom(double zet, double ekin) = 0; // for filling up the atomic loss

private:
  double ConvertLengthToKineticEnergy(double cutlenght);
  double ConvertKineticEnergyToLength(double cutenergy);

protected:
  int fParticleIndx; // 0 -> gamma, 1->e-, 2->e+
  int fMaxZet;       // maximum atomic number used in the geometry
  // define logaritmic spacing energy grid: can be set before initialisation
  int fNumEBins; // number of energy bins between fMinEnergy and fMaxEnergy; def 300
  double fMinCutEnergy;
  double fMaxCutEnergy;

  double *fEnergyGrid;   // size will be fNumEBins
  double *fLengthVector; // size if fNumEBins; for a given material, the length (range or absorption length) over the
                         // fEnergyGrid

  double **fElossOrAbsXsecTable; // per element; only for used so size will be fMaxZet; each element will point to
                                 // an array of dEdx or absorption cross section with size of fNumBins over the
                                 // fEnergyGrid
  int fMaxLengthIndx;
};

} // namespace GEANT_IMPL_NAMESPACE                                                                                       
} // namespace geantx
