#pragma once

#include "Geant/track/TrackState.hpp"
#include "Geant/track/TrackAccessor.hpp"
#include "Geant/track/TrackPhysicsAccessor.hpp"
#include "Geant/track/TrackCollection.hpp"

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
  {
  }

  void UpdateTime() const
  {
    PhysicsAccessor_t phys(fState, fParDef);
    fState.fTime += phys.E() * phys.Step() / phys.P();
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
    fState.fPhysicsState.fMomentum =
        std::sqrt(newKineticEnergy * (newKineticEnergy + 2 * fParDef.Mass()));

    TrackHistoryState &hist = fState.fHistoryState;
    // Increment generation
    ++hist.fGeneration;
    // Set mother ID and new ID
    hist.fMother   = other.Id();
    hist.fParticle = newId;
    // Set status (TODO: should this reset anything???)
    fState.fStatus = kNew;
  }
};

} // namespace geantx
