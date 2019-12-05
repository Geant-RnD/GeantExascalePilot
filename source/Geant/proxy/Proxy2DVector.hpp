//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/Proxy2DVector.hpp
 * @brief 
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/proxy/ProxyConstants.hpp"

namespace geantx {

class Proxy2DVector 
{
public:
  GEANT_HOST_DEVICE
  Proxy2DVector();

  GEANT_HOST_DEVICE
  ~Proxy2DVector(){};

  GEANT_HOST_DEVICE
  double Value(double x, double y);

  GEANT_HOST_DEVICE
  void PutX(size_t idx, double val);

  GEANT_HOST_DEVICE
  void PutY(size_t idy, double val);

  GEANT_HOST_DEVICE
  void PutValue(size_t idx, size_t idy, double val);

  GEANT_HOST_DEVICE
  double GetValue(size_t idx, size_t idy);

  GEANT_HOST_DEVICE
  size_t FindBinLocationX(double x);

  GEANT_HOST_DEVICE
  size_t FindBinLocationY(double y);

private:
  double xVector[brem::numberOfXNodes];
  double yVector[brem::numberOfYNodes];
  double value[brem::numberOfYNodes][brem::numberOfXNodes];
};

} // namespace geantx

