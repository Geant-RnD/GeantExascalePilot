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
 * @file Geant/proxy/ProxyElementUtils.cuh
 * @brief utility functions for Element
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/Config.hpp"

namespace geantx 
{
namespace ProxyElementUtils 
{
  GEANT_HOST_DEVICE
  double ComputeCoulombFactor(double zeff);

  GEANT_HOST_DEVICE
  double ComputeLradTsaiFactor(double zeff);

} // namespace ProxyElementUtils
} // namespace geantx

