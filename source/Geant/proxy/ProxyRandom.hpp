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

#include <VecMath/Rng.h>
#include <VecCore/VecCore>

namespace geantx {

class ProxyRandom {
public:
  ProxyRandom()
  {
    void *buff     = vecCore::AlignedAlloc(32, sizeof(vecRng::MRG32k3a<vecCore::backend::Scalar>));
    mrg32k3a = new (buff) vecRng::MRG32k3a<vecCore::backend::Scalar>;
    mrg32k3a->Initialize();
  }

  ~ProxyRandom()
  {
    mrg32k3a->~MRG32k3a();
    vecCore::AlignedFree(mrg32k3a);
  }

  double uniform() { return mrg32k3a->Uniform<vecCore::backend::Scalar>(); }

  double uniform(double a, double b) { return a + (b - a) * uniform(); }

  void uniform_array(size_t n, double *array, const double min = 0., const double max = 1.)
  {
    for (size_t i = 0; i < n; ++i) {
      array[i] = uniform(min, max);
    }
  }

  double Gauss(double mean, double sigma) { return mrg32k3a->Gauss<vecCore::backend::Scalar>(mean, sigma); }

private:
  vecRng::MRG32k3a<vecCore::backend::Scalar> *mrg32k3a;
};

} // namespace geantx
