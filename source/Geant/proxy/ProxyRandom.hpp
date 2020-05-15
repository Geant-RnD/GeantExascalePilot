//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyRandom.hpp
 * @brief A temporary implementation of Random
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/random/MRG32k3a.hpp"

namespace geantx {

class ProxyRandom {
public:

  ProxyRandom()
  {
    mrg32k3a = new MRG32k3a();
    mrg32k3a->Initialize();
  }

  ~ProxyRandom()
  {
    delete mrg32k3a;
  }

  GEANT_HOST_DEVICE
  double uniform() { return mrg32k3a->Uniform(); }

  GEANT_HOST_DEVICE
  double uniform(double a, double b) { return a + (b - a) * uniform(); }

  GEANT_HOST_DEVICE
  void uniform_array(size_t n, double *array, const double min = 0., const double max = 1.)
  {
    for (size_t i = 0; i < n; ++i) {
      array[i] = uniform(min, max);
    }
  }

private:
  MRG32k3a *mrg32k3a;
};

} // namespace geantx
