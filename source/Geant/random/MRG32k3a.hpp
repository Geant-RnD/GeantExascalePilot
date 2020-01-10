//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/random/MRG32k3a.hpp
 * @brief A SIMT implementation of MRG32k3a based on RngStream.h(cpp)
 *
 * RngStream is a class generating multiple streams of random numbers created
 * by Prof. Pierre L'Ecuyer, University of Montreal (lecuyer@iro.umontreal.ca)
 * Original source codes of RngStream.h(cpp) is available at
 * http://www.iro.umontreal.ca/~lecuyer/myftp/streams00/c++/
 *
 * Relevant articles in which MRG32k3a and the package with multiple streams
 * were proposed:
 *
 * P. L'Ecuyer, ``Good Parameter Sets for Combined Multiple Recursive Random
 * Number Generators'', Operations Research, 47, 1 (1999), 159--164.
 *
 * P. L'Ecuyer, R. Simard, E. J. Chen, and W. D. Kelton, ``An Objected-Oriented
 * Random-Number Package with Many Long Streams and Substreams'', Operations
 * Research, 50, 6 (2002), 1073--1075
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/random/MRG.hpp"
#include "Geant/random/RNG.hpp"

#include <cstring>
#include <iostream>

namespace geantx {

class MRG32k3a;

// struct MRG32k3a (random state of MRG32k3a)

template <> struct RNG_traits<MRG32k3a> {
  struct State {
    double fCg[MRG::vsize];
  };
  using State_t = State;
};

class MRG32k3a : public RNG<MRG32k3a> {

public:
  using State_t = typename RNG_traits<MRG32k3a>::State_t;

private:
  static double fSeed[MRG::vsize];
  double fBg[MRG::vsize];

  // Information on a stream: The arrays {Cg, Bg, Ig} (from the RngStream)
  // contain the current state of the stream, the starting state of the current
  // SubStream, and the starting state of the stream (not used in this class).
  // The next seed will be the seed of the next declared RngStream.

public:
  GEANT_HOST_DEVICE
  MRG32k3a() {}

  GEANT_HOST_DEVICE

  MRG32k3a(State_t *states) : RNG<MRG32k3a>(states) {}

  GEANT_HOST_DEVICE
  ~MRG32k3a() {}

  GEANT_HOST_DEVICE
  MRG32k3a(const MRG32k3a &rng);

  // Mandatory methods - static inheritance

  // Default initialization - automatic skipping to the next stream
  GEANT_HOST
  void Initialize() { SetNextStream(); }

  // Initialize with a unique stream number
  GEANT_HOST
  void Initialize(long streamId);

  // Initialize a set of states of which size is nthreads
  GEANT_HOST
  void Initialize(State_t *states, unsigned int nthreads);

  // Returns pRNG between 0 and 1
  GEANT_HOST_DEVICE
  double Kernel(State_t &state);

// Auxiliary methods

  GEANT_HOST
  void AdvanceState(long n, long e);

  GEANT_HOST void PrintState() const {
    return PrintState(*(this->fState));
  }

  GEANT_HOST void PrintState(State_t const &state) const;

  GEANT_HOST_DEVICE
  void SetSeed(double seed[MRG::vsize]);

private:
  // the mother is friend of this
  friend class RNG<MRG32k3a>;

  // Set Stream to NextStream/NextSubStream.
  GEANT_HOST
  void SetNextStream();

  GEANT_HOST
  void SetNextSubstream();

  // MRG32k3a utility methods
  GEANT_HOST
  double MultModM(double a, double s, double c, double m);

  GEANT_HOST
  void MatVecModM(const double A[3][3], const double s[3], double v[3],
                  double m);

  GEANT_HOST
  void MatTwoPowModM(const double A[3][3], double B[3][3], double m, long e);

  GEANT_HOST
  void MatPowModM(const double A[3][3], double B[3][3], double m, long n);

  GEANT_HOST
  void MatMatModM(const double A[3][3], const double B[3][3], double C[3][3],
                  double m);

  GEANT_HOST
  void TransitionVector(double C1[3][3], double C2[3][3], double e, double c);
};

} // namespace geantx

