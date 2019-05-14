#pragma once

#include "Geant/track/Types.hpp"
#include "Geant/track/TrackState.hpp"

namespace geantx {

//---------------------------------------------------------------------------//
/*!
 * \class TrackAccessor
 * \brief Common attributes about the track's state.
 */
class TrackAccessor {
  const TrackState &fTrack;

public:
  explicit TrackAccessor(const TrackState &track) : fTrack(track) {}

  // >>> ACCESSORS

  const Vector3D_t &Position() const { return this->Track().fPos; }
  const Vector3D_t &Direction() const { return this->Track().fDir; }
  double Step() const { return this->Track().fStep; }

  ParticleId_t Id() const { return this->Track().fHistoryState.fParticle; }

  // TODO: treat as protected/implementation detail? used by TrackModifier
  const TrackState &Track() const { return fTrack; }
};

} // namespace geantx
