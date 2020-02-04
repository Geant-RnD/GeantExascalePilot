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
 * @brief Use to modify the state of a Track.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackAccessor.hpp"
#include "Geant/track/TrackCollection.hpp"
#include "Geant/track/TrackPhysicsAccessor.hpp"
#include "Geant/track/TrackState.hpp"

namespace geantx {
template <typename PD>
class TrackModifier {
  using Base = TrackAccessor;

public:
  using ParticleDef_t     = PD;
  using TrackId_t         = TrackCollection::TrackId_t;
  using PhysicsAccessor_t = TrackPhysicsAccessor<ParticleDef_t>;

private:
  TrackState &fState;
  const ParticleDef_t &fParDef;

public:
  TrackModifier(TrackCollection &tracks, TrackId_t track_id,
                const ParticleDefinitions &pdefs)
      : fState(tracks.Get(track_id)),
        fParDef(PhysicsAccessor_t::GetParticleDef(pdefs, fState))
  {}

  //! Move in a straight line
  void Step(double distance)
  {
    REQUIRE(distance <= fState.fGeometryState.fSnext);

    // Update particle position
    fState.fPos += distance * fState.fDir;
    // Decrement straight-line distance-to-boundary
    fState.fGeometryState.fSnext -= distance;

    return this->StepImpl(distance);
  }

  //! Move in a curved path
  void Step(double distance, const ThreeVector &new_position,
            const ThreeVector &new_direction)
  {
    REQUIRE(distance > Norm(new_position - fState.fPos));
    REQUIRE(IsUnitVector(new_direction));

    fState.fPos = new_position;
    fState.fDir = new_direction;
    // XXX: how to update safety/snext?

    return this->StepImpl(distance);
  }

  //! Collision: change direction and energy
  void Collide(const ThreeVector &new_direction, double newKineticEnergy) const
  {
    REQUIRE(newKineticEnergy >= 0);
    REQUIRE(SoftUnitVector(newDirection));

    fState.fDir                = new_direction;
    fState.fPhysicsState.fEkin = newKineticEnergy;

    // Update momentum
    fState.fPhysicsState.fMomentum =
        std::sqrt(newKineticEnergy * (newKineticEnergy + 2 * fParDef.Mass()));
  }

  //! Multiple scattering collision: just change direction
  void Collide(const ThreeVector &new_direction) const
  {
    REQUIRE(SoftUnitVector(newDirection));

    fState.fDir = new_direction;
    // XXX Invalidate distance-to-boundary
  }

  //! Kill the particle
  void Kill() const
  {
    fState.fStatus = TrackStatus::Killed;
    // XXX set ekin to zero?
  }

  template <typename OPD>
  void CreateSecondaryFrom(const TrackPhysicsAccessor<OPD> &other, ParticleId_t newId,
                           const ThreeVector &newDirection, double newKineticEnergy) const
  {
    REQUIRE(newId > TrackAccessor(other).Id());
    REQUIRE(newKineticEnergy >= 0);
    REQUIRE(SoftUnitVector(newDirection));

    // Copy most properties (using implementation detail: subject to change)
    fState = other.State();

    // Set new physical properties
    fState.fPhysicsState.fParticleDefId = fParDef.Id();
    this->Collide(newDirection, newKineticEnergy);

    TrackHistoryState &hist = fState.fHistoryState;
    // Increment generation
    ++hist.fGeneration;
    // Set mother ID and new ID
    hist.fMother   = other.Id();
    hist.fParticle = newId;
    // Set status (TODO: should this reset anything???)
    fState.fStatus = TrackStatus::New;
  }

private:
  void StepImpl(double step) const
  {
    // Decrement distance-to-physics
    fState.fPhysicsState.fPstep -= step;
    // Increment step counter
    ++fState.fHistoryState.fNsteps;

    // Update time XXX is this in the right place? (before/after energy loss?)
    PhysicsAccessor_t phys(fState, fParDef);
    fState.fTime += phys.E() * step / phys.P();
  }
};

} // namespace geantx
