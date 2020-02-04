//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyPhysicsVector.hpp
 * @brief
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"

namespace geantx {

class ProxyPhysicsVector {

public:
  GEANT_HOST_DEVICE
  ProxyPhysicsVector();

  GEANT_HOST_DEVICE
  ~ProxyPhysicsVector();

  GEANT_HOST_DEVICE
  ProxyPhysicsVector(ProxyPhysicsVector &right);

  GEANT_HOST
  void Allocate(int nsize);

  GEANT_HOST_DEVICE
  void Deallocate();

  GEANT_HOST_DEVICE
  int SizeOfVector();

  GEANT_HOST
  bool Retrieve(std::ifstream &fIn);

  GEANT_HOST
  void Relocate(void *devPtr);

  GEANT_HOST_DEVICE
  double Value(double energy);

  GEANT_HOST_DEVICE
  double Energy(size_t index);

  GEANT_HOST_DEVICE
  bool SplinePossible();

  GEANT_HOST_DEVICE
  void ComputeSecDerivatives();

  GEANT_HOST_DEVICE
  void ComputeSecondDerivatives(double firstPointDerivative, double endPointDerivative);

  GEANT_HOST_DEVICE
  void FillSecondDerivatives();

  GEANT_HOST_DEVICE
  int FindBinLocation(double energy);

  GEANT_HOST_DEVICE
  double SplineInterpolation(double energy, int lastBin);

  GEANT_HOST_DEVICE
  double LinearInterpolation(double energy, int lastBin);

  GEANT_HOST_DEVICE
  double Interpolation(double energy, int lastBin);

  GEANT_HOST_DEVICE
  void SetSpline(bool val);

  // special treatment for the Inverse Range Table (G4LPhysicsFreeVector)
  GEANT_HOST_DEVICE
  double InvRangeValue(double energy);

  GEANT_HOST_DEVICE
  int InvRangeFindBinLocation(double energy);

  // accessors
  GEANT_HOST_DEVICE
  inline void SetType(int type) { fType = type; }

private:
  int fType; // The type of PhysicsVector
  bool fUseSpline;
  bool fIsSecondDerivativeFilled;
  int fNumberOfNodes;

  double fDeltaBin; // Bin width - useful only for fixed binning
  double fBaseBin;  // Set this in constructor for performance
  double fEdgeMin;  // Energy of first point
  double fEdgeMax;  // Energy of the last point

  double *fDataVector;
  double *fBinVector;
  double *fSecDerivative;
};

} // namespace geantx
