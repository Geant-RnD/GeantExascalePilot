#pragma once

#include <cstdint>
#include <vector>
#include "Geant/track/TrackState.hpp"

namespace geantx {

class TrackCollection {
public:
  using size_type = std::size_t;
  using TrackId_t = size_type;

private:
  std::vector<TrackState> fTracks;

public:


private:
  using value_type = TrackState;

  // FIXME: temporary implementation
  const value_type& Get(TrackId_t i) const { return fTracks.at(i); }
  value_type& Get(TrackId_t i) { return fTracks.at(i); }

  template<class PT> friend class TrackModifier;
  friend class TrackAccessor;
};

} // namespace geantx

