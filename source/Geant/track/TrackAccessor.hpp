#pragma once

#include "Geant/track/Types.hpp"
#include "Geant/track/TrackCollection.hpp"
#include "Geant/track/TrackState.hpp"

namespace geantx {

//---------------------------------------------------------------------------//
/*!
 * \class TrackAccessor
 * \brief Common attributes about the track's state.
 */
class TrackAccessor {
  const TrackState &fState;

public:
  using TrackId_t = TrackCollection::TrackId_t;

public:
  TrackAccessor(const TrackCollection &tracks, TrackId_t track_id)
      : fState(tracks.Get(track_id))
  {
  }

  // >>> ACCESSORS

  const ThreeVector &Position() const { return this->State().fPos; }
  const ThreeVector &Direction() const { return this->State().fDir; }
  double Step() const { return this->State().fStep; }

  ParticleId_t Id() const { return this->State().fHistoryState.fParticle; }

protected:
  // >>> IMPLEMENTATION DETAILS

  explicit TrackAccessor(const TrackState &state) : fState(state) {}
  const TrackState &State() const { return fState; }

  template <typename PT>
  friend class TrackModifier;
};

} // namespace geantx
