#pragma once

#include "Track.hpp"

//---------------------------------------------------------------------------//
/*!
 * \class TrackPhysicsAccessor
 * \brief Abstract physics state information access for a track.
 *
 * This currently takes just a track state, but if the data layout changes, it
 * could take e.g. some abstract multi-particle state *plus* a particle index.
 */
class TrackPhysicsAccessor : public TrackAccessor {
    using Base = TrackAccessor;
public:
  explicit TrackPhysicsAccessor(const TrackState &track) : Base(track) {}

  /* EXAMPLE alternate constructor:
   *
   TrackPhysicsAccessor(const TrackPool &pool, int threadIdx) : fTrack(pool[threadIdx]) {}
   */

  // >>> ACCESSORS

  double Mass() const { return this->Pstate().fMass; }
  double Charge() const { return this->Pstate().fCharge; }
  // TODO: replace some of these with calculated quantities?
  double P() const { return this->Pstate().fMomentum; }
  double E() const { return this->Pstate().fEnergy; }
  double LogEkin() const { return this->Pstate().fLogEkin; }
  double Ekin() const { return this->Pstate().fE - this->Pstate().mass; }

  MaterialId_t Material() const { return this->Pstate().fMaterial; }

private:
  inline const PhysicsState &Pstate() const { return fTrack.fPhysicsState; }
};

