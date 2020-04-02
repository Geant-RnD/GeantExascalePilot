//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/random/RNG.hpp
 * @brief The base class of random number generators 
 *
 * Requirements :
 * 1) T  : A pseudo-random number generator with multiple streams
 * 2) State_t   : A templated struct of T states
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"

namespace geantx {

template <typename T> struct RNG_traits;

template <typename T> class RNG {

protected:
  // Use *this to access data members in the derived class
  using State_t = typename RNG_traits<T>::State_t;
  State_t *fState;

public:
  GEANT_HOST_DEVICE
  RNG() { fState = new State_t; }

  // Dummy Constructor for SIMT
  GEANT_HOST_DEVICE
  RNG(State_t *devState) { fState = devState; }

  GEANT_HOST_DEVICE
  ~RNG() { delete fState; }

  RNG(const RNG &rng) = default;

  // Static interfaces (Required methods)

  // Initialization for SIMD
  GEANT_HOST
  void Initialize() { static_cast<T *>(this)->template Initialize(); }

  // Initialization with a unique stream number
  GEANT_HOST
  void Initialize(long streamId) {
    static_cast<T *>(this)->template Initialize(streamId);
  }

  // Initialization for SIMT
  GEANT_HOST
  void Initialize(State_t *states, unsigned int nthreads) {
    static_cast<T *>(this)->template Initialize(states, nthreads);
  }

  // Return BackendT::Double_v of random numbers in [0,1)
  GEANT_HOST_DEVICE 
  double Uniform() {
    return static_cast<T *>(this)->Kernel(*this->fState);
  }

  // Generate random numbers based on a given state
  GEANT_HOST_DEVICE
  double Uniform(State_t *state) {
    return static_cast<T *>(this)->Kernel(*state);
  }

  GEANT_HOST
  void PrintState() const { static_cast<T *>(this)->PrintState(); }

  // Auxiliary methods

  GEANT_HOST_DEVICE
  void SetState(State_t *state) { fState = state; }

  GEANT_HOST_DEVICE
  State_t *GetState() const { return fState; }

  GEANT_HOST_DEVICE
  State_t const &GetStateRef() const { return *fState; }

  // Common methods

  // UniformIndex 
  GEANT_HOST_DEVICE
  long UniformIndex(long min = 0, long max = UINT64_MAX) {
    return min + (max - min) * static_cast<T *>(this)->Uniform();
  }

  // UniformIndex with a status
  GEANT_HOST_DEVICE
  long UniformIndex(State_t *state, long min = 0, long max = UINT64_MAX) {
    return min + (max - min) * static_cast<T *>(this)->Uniform(state);
  }

  // Returns an array of random numbers
  GEANT_HOST_DEVICE
  void Array(const size_t nsize, double *array);

  // Flat distribution in [min,max)
  GEANT_HOST_DEVICE
  double Flat(double min, double max) {
    return min + (max - min) * static_cast<T *>(this)->Uniform();
  }

  // Flat distribution in [min,max] with a state
  GEANT_HOST_DEVICE
  double Flat(State_t *state, double min,double max) {
    return min + (max - min) * static_cast<T *>(this)->Uniform(state);
  }

  // Exponential deviates: exp(-x/tau)
  GEANT_HOST_DEVICE
  double Exp(double tau);

  // Exponential deviates with a state
  GEANT_HOST_DEVICE
  double Exp(State_t *state, double tau);

};

// Implementation
// Common Methods
// Returns an array of random numbers
template <typename T>
GEANT_HOST_DEVICE
void RNG<T>::Array(const size_t nsize, double  *array) {
  for (size_t i = 0; i < nsize; ++i) {
    double u01 = static_cast<T *>(this)->template Uniform();
    array[i] = u01;
  }
}

// Exponential deviates: exp(-x/tau)
template <typename T>
GEANT_HOST_DEVICE
double RNG<T>::Exp(double tau) {
  double u01 = static_cast<T *>(this)->Uniform();
  //@syj: check for zero
  return -tau * log(u01);
}

// Exponential deviates with a state
template <typename T>
GEANT_HOST_DEVICE
double RNG<T>::Exp(State_t *state, double tau) {
  // Exp with a state
  double u01 = static_cast<T *>(this)->Uniform(state);
  return -tau * log(u01);
}

} // namespace geantx

