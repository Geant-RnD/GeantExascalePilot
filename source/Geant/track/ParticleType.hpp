#pragma once

// These data structures could be put in constant memory on GPU for
// register-speed performance, or we could template or something

#include "Geant/track/Types.hpp"

struct ParticleType {
  double fCharge;
  double fRestMass;
  int fCode; /* Particle Data Group code index */
  Species_t fSpecies;
};
