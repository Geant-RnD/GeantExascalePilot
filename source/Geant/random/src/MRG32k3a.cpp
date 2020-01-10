//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/random/src/MRG32k3a.cpp
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

#include "Geant/random/MRG32k3a.hpp"

namespace geantx {

// The default seed of MRG32k3a
double MRG32k3a::fSeed[MRG::vsize] = {12345., 12345., 12345., 12345., 12345., 12345.};

//
// Class Implementation
//

// Copy constructor
GEANT_HOST_DEVICE
MRG32k3a::MRG32k3a(const MRG32k3a &rng) : RNG<MRG32k3a>() {
  for (int i = 0; i < MRG::vsize; ++i) {
    this->fState->fCg[i] = rng.fState->fCg[i];
    fSeed[i] = rng.fSeed[i];
    fBg[i] = rng.fBg[i];
  }
}

// Reset stream to the next Stream.
GEANT_HOST void MRG32k3a::SetNextStream() {
  for (int i = 0; i < MRG::vsize; ++i) {
    this->fState->fCg[i] = fBg[i] = fSeed[i];
  }
  MatVecModM(MRG::A1p127, fSeed, fSeed, MRG::m1);
  MatVecModM(MRG::A2p127, &fSeed[3], &fSeed[3], MRG::m2);
}

// Scalar specialization of SetNextSubstream
void MRG32k3a::SetNextSubstream() {
  for (int i = 0; i < MRG::vsize; ++i) {
    this->fState->fCg[i] = fBg[i];
  }
  MatVecModM(MRG::A1p76, fBg, fBg, MRG::m1);
  MatVecModM(MRG::A2p76, &fBg[3], &fBg[3], MRG::m2);
}

GEANT_HOST
void MRG32k3a::Initialize(long streamId) {
  // start from the default state
  for (int i = 0; i < MRG::vsize; ++i)
    fSeed[i] = 12345.;

  // reset the state to the biginning of the first stream
  Initialize();

  // skip-ahead by (the stream number)*(the size of stream length in powers of
  // 2)
  long e = streamId * MRG::slength;
  AdvanceState(e, 0);
}

// Specialization for the scalar backend to initialize an arrary of states of
// which size is threads. "states" should be allocated beforehand, but can be
// used for both host and device pointers
GEANT_HOST
void MRG32k3a::Initialize(State_t *states, unsigned int nthreads) {
  State_t *hstates = (State_t *)malloc(nthreads * sizeof(State_t));

  for (unsigned int tid = 0; tid < nthreads; ++tid) {
    SetNextStream();
    for (size_t j = 0; j < MRG::vsize; ++j) {
      hstates[tid].fCg[j] = this->fState->fCg[j];
    }
  }
#ifdef VECCORE_CUDA
  cudaMemcpy(states, hstates, nthreads * sizeof(State_t),
             cudaMemcpyHostToDevice);
#else
  memcpy(states, hstates, nthreads * sizeof(State_t));
#endif
  free(hstates);
}

// Print information of the current state
GEANT_HOST
void MRG32k3a::PrintState(State_t const &state) const {
  for (size_t j = 0; j < MRG::vsize; ++j) {
    std::cout << state.fCg[j] << " ";
  }
  std::cout << std::endl;
}

// Set the next seed
GEANT_HOST_DEVICE
void MRG32k3a::SetSeed(double seed[MRG::vsize]) {
  for (int i = 0; i < MRG::vsize; ++i)
    fSeed[i] = seed[i];
}

// Kernel to generate the next random number(s) based on RngStream::U01d
GEANT_HOST_DEVICE
double MRG32k3a::Kernel(State_t &state) {
  double k, p1, p2;

  // Component 1
  p1 = MRG::a12 * state.fCg[1] - MRG::a13n * state.fCg[0];
#if __CUDA_ARCH__ > 0
  k = trunc(fma(p1, MRG::rh1, p1 * MRG::rl1));
#else
  k = floor(p1 / MRG::m1);
#endif
  p1 -= k * MRG::m1;

  if (p1 < 0.) p1 += MRG::m1;

  state.fCg[0] = state.fCg[1];
  state.fCg[1] = state.fCg[2];
  state.fCg[2] = p1;

  p2 = MRG::a21 * state.fCg[5] - MRG::a23n * state.fCg[3];
#if __CUDA_ARCH__ > 0
  k = trunc(fma(p2, MRG::rh2, p2 * MRG::rl2));
#else
  k = floor(p2 / MRG::m2);
#endif
  p2 -= k * MRG::m2;

  if (p2 < 0.) p2 += MRG::m2;

  state.fCg[3] = state.fCg[4];
  state.fCg[4] = state.fCg[5];
  state.fCg[5] = p2;

  // Combination
  return (p1 > p2) ? (p1 - p2) * MRG::norm : (p1 - p2 + MRG::m1) * MRG::norm;

  // Extended (53 bits) precision
  // Double_v random =  Blend((p1 > p2),(p1 - p2) * MRG::norm, (p1 - p2 +
  // MRG::m1) * MRG::norm); random *= (1.0 + MRG::fact); return Blend((random
  // < 1.0), random, random - 1.0);
}

// Utility functions from RngSteam

// if e > 0, let n = 2^e + c;
// if e < 0, let n = -2^(-e) + c;
// if e = 0, let n = c.
// Jump n steps forward if n > 0, backwards if n < 0.
void MRG32k3a::AdvanceState(long e, long c) {
  double C1[3][3], C2[3][3];

  TransitionVector(C1, C2, e, c);

  MatVecModM(C1, this->fState->fCg, this->fState->fCg, MRG::m1);
  MatVecModM(C2, &(this->fState->fCg[3]), &(this->fState->fCg[3]), MRG::m2);
}

GEANT_HOST
void MRG32k3a::TransitionVector(double C1[3][3], double C2[3][3],
                                          double e, double c) {
  double B1[3][3], B2[3][3];

  if (e > 0) {
    MatTwoPowModM(MRG::A1p0, B1, MRG::m1, e);
    MatTwoPowModM(MRG::A2p0, B2, MRG::m2, e);
  } else if (e < 0) {
    MatTwoPowModM(MRG::InvA1, B1, MRG::m1, -e);
    MatTwoPowModM(MRG::InvA2, B2, MRG::m2, -e);
  }

  if (c >= 0) {
    MatPowModM(MRG::A1p0, C1, MRG::m1, c);
    MatPowModM(MRG::A2p0, C2, MRG::m2, c);
  } else {
    MatPowModM(MRG::InvA1, C1, MRG::m1, -c);
    MatPowModM(MRG::InvA2, C2, MRG::m2, -c);
  }

  if (e) {
    MatMatModM(B1, C1, C1, MRG::m1);
    MatMatModM(B2, C2, C2, MRG::m2);
  }
}

// Return (a*s + c) MOD m; a, s, c and m must be < 2^35
GEANT_HOST
double MRG32k3a::MultModM(double a, double s, double c, double m) {
  double v;
  long a1;

  v = a * s + c;

  if (v >= MRG::two53 || v <= -MRG::two53) {
    a1 = static_cast<long>(a / MRG::two17);
    a -= a1 * MRG::two17;
    v = a1 * s;
    a1 = static_cast<long>(v / m);
    v -= a1 * m;
    v = v * MRG::two17 + a * s + c;
  }

  a1 = static_cast<long>(v / m);
  // in case v < 0)
  if ((v -= a1 * m) < 0.0)
    return v += m;
  else
    return v;
}

// Compute the vector v = A*s MOD m. Assume that -m < s[i] < m. Works also when
// v = s.
GEANT_HOST
void MRG32k3a::MatVecModM(const double A[3][3], const double s[3],
                                    double v[3], double m) {
  int i;
  // Necessary if v = s
  double x[3];

  for (i = 0; i < 3; ++i) {
    x[i] = MultModM(A[i][0], s[0], 0.0, m);
    x[i] = MultModM(A[i][1], s[1], x[i], m);
    x[i] = MultModM(A[i][2], s[2], x[i], m);
  }
  for (i = 0; i < 3; ++i)
    v[i] = x[i];
}

// Compute the matrix C = A*B MOD m. Assume that -m < s[i] < m.
// Note: works also if A = C or B = C or A = B = C.
GEANT_HOST
void MRG32k3a::MatMatModM(const double A[3][3], const double B[3][3],
                                    double C[3][3], double m) {
  int i, j;
  double V[3], W[3][3];

  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j)
      V[j] = B[j][i];
    MatVecModM(A, V, V, m);
    for (j = 0; j < 3; ++j)
      W[j][i] = V[j];
  }
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j)
      C[i][j] = W[i][j];
  }
}

// Compute the matrix B = (A^(2^e) Mod m);  works also if A = B.
GEANT_HOST
void MRG32k3a::MatTwoPowModM(const double A[3][3], double B[3][3],
                                       double m, long e) {
  int i, j;

  // Initialize: B = A
  if (A != B) {
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j)
        B[i][j] = A[i][j];
    }
  }
  // Compute B = A^(2^e) mod m
  for (i = 0; i < e; i++)
    MatMatModM(B, B, B, m);
}

// Compute the matrix B = (A^n Mod m);  works even if A = B.
GEANT_HOST
void MRG32k3a::MatPowModM(const double A[3][3], double B[3][3],
                                    double m, long n) {
  int i, j;
  double W[3][3];

  // initialize: W = A; B = I
  for (i = 0; i < 3; ++i) {
    for (j = 0; j < 3; ++j) {
      W[i][j] = A[i][j];
      B[i][j] = 0.0;
    }
  }
  for (j = 0; j < 3; ++j)
    B[j][j] = 1.0;

  // Compute B = A^n mod m using the binary decomposition of n
  while (n > 0) {
    if (n % 2)
      MatMatModM(W, B, B, m);
    MatMatModM(W, W, W, m);
    n /= 2;
  }
}

} // namespace geantx

