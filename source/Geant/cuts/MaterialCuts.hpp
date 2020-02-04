//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/cuts/MaterialCuts.hpp
 * @brief Originated in the GeantV project
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>
#include <iostream>

#include "Geant/material/Material.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

class Material;

// class Material;
/**
 * @brief   Material - particle production cut object (dummy class at the moment).
 * @class   MaterialCuts
 * @author  M Novak, A Ribon
 * @date    january 2016
 *
 * Object to store partcile production cuts in material. Production cuts are currently
 * available for gamma, electron and positron particles and stored both in energy
 * threshold and length. Each constructed material - cuts object pointer is stored in a
 * global table that can be obtained through the static
 * MaterialCuts::GetTheMaterialCutsTable method. Each matrial-cuts object stores its own
 * index in the global table that can be obtained by MaterialCuts::GetIndex method. Each
 * matrial-cuts object stores a pointer to the material object (does not own the material
 * object) that can be obtained by MaterialCuts::GetMaterial.
 *
 * Production cuts in energy/lenght are stored in arrays index by particle type:
 * \begin{listings}
 *  \mathrm{index} = 0  & \mathrm{gamma particle} \\
 *  \mathrm{index} = 1  & \mathrm{electron}       \\
 *  \mathrm{index} = 2  & \mathrm{positron}
 * \end{listings}
 *
 * The energy/length production cuts arrays can be obtained by
 * MaterialCuts::GetProductionCutsInEnergy/MaterialCuts::GetProductionCutsInLength.
 * Production cuts in energy/length for a given particle can be set by
 * MaterialCuts::SetProductionCutEnergy/MaterialCuts::SetProductionCutLength by providing
 * the particle index (see above) and the appropriate production cut in appropriate
 * internal units.
 *
 *
 * \todo: this is a very basic implementation with the minimal functionality. It will be
 * developed further together with the general physics framework and functionalities like
 * converting production cuts from length to energy threshold or possibility to set
 * production cuts either in energy threshold or length will be added. More detailed
 * documentation will be provided later with the implementation of these new
 * functionalities.
 */
class MaterialCuts {
public:
  // creates all MaterialCuts and converts length/energy to energy/lenght
  static void CreateAll();
  // deletes all MaterialCuts objects and set the table to default (zero) size
  static void ClearAll();

  /** @brief Public method to obtain the index of this material-cuts object in the global
   * table. */
  int GetIndex() const { return fIndex; }

  const double *GetProductionCutsInLength() const { return fProductionCutsInLength; }
  const double *GetProductionCutsInEnergy() const { return fProductionCutsInEnergy; }
  bool IsProductionCutsGivenInLength() const { return fIsProductionCutsGivenInLength; }

  const Material *GetMaterial() const { return fMaterial; }
  // get a MaterialCuts object pointer by its index
  static const MaterialCuts *GetMaterialCut(int indx);

  // get the global mategrial-cuts table
  static const std::vector<MaterialCuts *> &GetTheMaterialCutsTable()
  {
    return gTheMaterialCutsTable;
  }

private:
  /**
   * @brief Dummy constructor for testing physics models. Will be removed.
   *
   * Production cuts are set to the provided values.
   *
   * @param[in] mat          Pointer to specify the material object part of this
   * mategrial-cuts pair.
   * @param[in] gcutlength   Production cut for gamma particle in length.
   * @param[in] emcutlength  Production cut for electron in length.
   * @param[in] epcutlength  Production cut for positron in length.
   * @param[in] gcutenergy   Production cut for gamma particle in energy.
   * @param[in] emcutenergy  Production cut for electron in energy.
   * @param[in] epcutenergy  Production cut for positron in energy.
   */
  MaterialCuts(int regionindx, const Material *mat, bool iscutinlength, double gcut,
               double emcut, double epcut);
  /** @brief Destructor */
  ~MaterialCuts(){};

  // NOTE: they might be private if we create all matcuts automatically
  // if not   // TODO: add check of indices
  void SetProductionCutEnergy(int indx, double val)
  {
    fProductionCutsInEnergy[indx] = val;
  }
  void SetProductionCutLength(int indx, double val)
  {
    fProductionCutsInLength[indx] = val;
  }

  // convert gamma,e-,e+ cuts length to energy or energy to length
  static void ConvertAll();

private:
  static const int kNumProdCuts = 3;
  int fIndex;                          // in the global table
  bool fIsProductionCutsGivenInLength; // is production cuts are given by the user in
                                       // length
  double fProductionCutsInLength[kNumProdCuts];
  double fProductionCutsInEnergy[kNumProdCuts];

  const Material *fMaterial; // does not own this material onject

  // the global mategrial-cuts table; filled constatntly when MaterialCuts obejcts are
  // created
  static std::vector<MaterialCuts *> gTheMaterialCutsTable;
};

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
