//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Static information about the particles.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackState.hpp"
#include <cstdint>
#include <vector>

namespace geantx {
// FIXME: placeholder
class ParticleDefinition {
public:
  double Charge() const { return 0; }
  ParticleSpecies Species() const { return ParticleSpecies::Hadron; }
  double Mass() const { return 1.0; }
};

// FIXME: interface is TBD
class ParticleDefinitions {
public:
  using size_type = std::size_t;
  using PDefId_t  = size_type;

private:
  std::vector<ParticleDefinition> fPdef;

private:
  using value_type = ParticleDefinition;

  // FIXME: temporary implementation
  const value_type &Get(PDefId_t i) const { return fPdef.at(i); }
  value_type &Get(PDefId_t i) { return fPdef.at(i); }

  template <typename PT>
  friend class TrackModifier;
  template <typename PT>
  friend class TrackPhysicsAccessor;
};

} // namespace geantx
