#pragma once

#include "Geant/track/TrackState.hpp"
#include "Geant/track/TrackCollection.hpp"
#include "Geant/track/TrackAccessor.hpp"

namespace geantx {

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
  explicit TrackGeometryAccessor(const TrackCollection &tracks, TrackId_t track_id) : Base(tracks, track_id) {}

  // >>> ACCESSORS

  VolumeId_t Volume() const { return this->Gstate().fVolume; }

private:
  const GeometryState &Gstate() const { return this->.fGeometryState; }
};

} // namespace geantx
