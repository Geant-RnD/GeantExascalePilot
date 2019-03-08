#pragma once

#include "TrackState.hpp"
#include "PhysicsAccessor.hpp"

class TrackModifier {
private:
  TrackState &fState;

public:
  explicit TrackModifier(TrackState &track) : fState(track) {}

  void UpdateTime() const
  {
    PhysicsAccessor phys(fState);
    fState.fTime += phys.E() * phys.Step() / phys.P();
  }

  void CreateSecondaryFrom(const TrackPhysicsAccessor &other,
                           ParticleType_t newType,
                           ParticleId_t newId,
                           const Vector3 &newDirection,
                           double newKineticEnergy) const {
      REQUIRE(newId > TrackAccessor(other).Id());
      REQUIRE(newKineticEnergy >= 0);
      REQUIRE(SoftUnitVector(newDirection));

      // Copy most properties
      fState = other.Track();

      // Set new physical properties
      fState.fPhysicsState.fParticleType = newType;
      fState.fMomentum = std::sqrt(newKineticEnergy * (newKineticEnergy
                                                       + 2 * other.Mass()));

      TrackHistoryState& hist = fState.fHistoryState;
      // Increment generation
      ++hist.fGeneration;
      // Set mother ID and new ID
      hist.fMother = other.Id();
      hist.fParticle = new_id;
      // Set status (TODO: should this reset anything???)
      hist.fStatus = kNew;
  }
};
