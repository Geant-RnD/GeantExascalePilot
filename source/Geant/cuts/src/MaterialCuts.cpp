//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/cuts/src/MaterialCuts.cpp
 * @brief Originated in the GeantV project
 */
//===----------------------------------------------------------------------===//

#include "Geant/cuts/MaterialCuts.hpp"

#include "Geant/material/Material.hpp"

#include "Geant/cuts/CutConverter.hpp"
#include "Geant/cuts/CutConverterForGamma.hpp"
#include "Geant/cuts/CutConverterForElectron.hpp"
#include "Geant/cuts/CutConverterForPositron.hpp"

#include "Geant/particles/PhysicsParameters.hpp"

#include "Geant/core/SystemOfUnits.hpp"

#include <iostream>

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

Vector_t<MaterialCuts *> MaterialCuts::gTheMaterialCutsTable;

MaterialCuts::MaterialCuts(int regionindx, const Material *mat, bool iscutinlength,
                           double gcut, double emcut, double epcut)
    : fIsProductionCutsGivenInLength(iscutinlength), fMaterial(mat)
{
  if (iscutinlength) {
    fProductionCutsInLength[0] = gcut;  // for gamma in lenght
    fProductionCutsInLength[1] = emcut; // for e- in lenght
    fProductionCutsInLength[2] = epcut; // for e+ in lenght
    // they will be set by the converter
    fProductionCutsInEnergy[0] = -1.0; // for gamma in internal energy units
    fProductionCutsInEnergy[1] = -1.0; // for e- in internal energy units
    fProductionCutsInEnergy[2] = -1.0; // for e+ in internal energy units
  } else {
    // they will be set by the converter
    fProductionCutsInLength[0] = -1.0; // for gamma in lenght
    fProductionCutsInLength[1] = -1.0; // for e- in lenght
    fProductionCutsInLength[2] = -1.0; // for e+ in lenght

    fProductionCutsInEnergy[0] = gcut;  // for gamma in internal energy units
    fProductionCutsInEnergy[1] = emcut; // for e- in internal energy units
    fProductionCutsInEnergy[2] = epcut; // for e+ in internal energy units
  }

  fIndex = gTheMaterialCutsTable.size();
  // add the current MaterialCuts pointer to the global material cuts table
  gTheMaterialCutsTable.push_back(this);
}

void MaterialCuts::ClearAll()
{
  for (unsigned long i = 0; i < gTheMaterialCutsTable.size(); ++i) {
    delete gTheMaterialCutsTable[i];
  }
  gTheMaterialCutsTable.clear();
}

const MaterialCuts *MaterialCuts::GetMaterialCut(int indx)
{
  if (indx > -1 && indx < int(gTheMaterialCutsTable.size())) {
    return gTheMaterialCutsTable[indx];
  } else {
    std::cerr << "  ***  ERROR:  MaterialCuts::GetMaterialCut() \n"
              << "        Requested MaterialCuts by index = " << indx
              << " cannot be find in the table! \n"
              << "        Index should be:   0 <= index < number of MaterialCuts = "
              << gTheMaterialCutsTable.size() << std::endl;
    exit(-1);
  }
}

void MaterialCuts::ConvertAll()
{
  CutConverter **converters = new CutConverter *[3];
  // get the min/max secondary production values; they are the same in each reagion
  converters[0] =
      new CutConverterForGamma(301, PhysicsParameters::GetMinAllowedGammaCutEnergy(),
                               PhysicsParameters::GetMaxAllowedGammaCutEnergy());
  converters[1] = new CutConverterForElectron(
      301, PhysicsParameters::GetMinAllowedElectronCutEnergy(),
      PhysicsParameters::GetMaxAllowedElectronCutEnergy());
  converters[2] = new CutConverterForPositron(
      301, PhysicsParameters::GetMinAllowedPositronCutEnergy(),
      PhysicsParameters::GetMaxAllowedPositronCutEnergy());

  for (unsigned long i = 0; i < gTheMaterialCutsTable.size(); ++i) {
    MaterialCuts *matcut = gTheMaterialCutsTable[i];
    const Material *mat  = matcut->GetMaterial();
    if (matcut->fIsProductionCutsGivenInLength) {
      for (int j = 0; j < 3; ++j) {
        const double *cuts = matcut->GetProductionCutsInLength();
        matcut->SetProductionCutEnergy(j, converters[j]->Convert(mat, cuts[j], true));
      }
    } else {
      for (int j = 0; j < 3; ++j) {
        const double *cuts = matcut->GetProductionCutsInEnergy();
        matcut->SetProductionCutLength(j, converters[j]->Convert(mat, cuts[j], false));
      }
    }
  }
  delete converters[0];
  delete converters[1];
  delete converters[2];
  delete[] converters;
}

// create all MaterialCuts by using the Region table; this will be the standard way of
// automatically creating all MaterialCuts in the detetector
void MaterialCuts::CreateAll()
{
  // clear all if there were any created before
  ClearAll();
  // set all Materials used flag to false
  const Vector_t<Material *> theMaterialTable = Material::GetTheMaterialTable();
  for (size_t i = 0; i < theMaterialTable.size(); ++i) {
    theMaterialTable[i]->SetIsUsed(false);
  }
  // convert production cuts in lenght/energy to energy/lenght
  ConvertAll();
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
