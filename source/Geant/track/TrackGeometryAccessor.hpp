#pragma once

#include "Track.hpp"
#include "TrackAccessor.hpp"

//---------------------------------------------------------------------------//
/*!
 * \class TrackGeometryAccessor
 * \brief Abstract geometry state information access for a track.
 *
 * This currently takes just a track state, but if the data layout changes, it
 * could take e.g. some abstract multi-particle state *plus* a particle index.
 */
class TrackGeometryAccessor : public TrackAccessor {
  using Base = TrackAccessor;

public:
  explicit TrackGeometryAccessor(const TrackState &track) : Base(track) {}

  // >>> ACCESSORS

  VolumeId_t Volume() const { return this->Gstate().fVolume; }

private:
  const GeometryState &Gstate() const { return this->.fGeometryState; }
};
