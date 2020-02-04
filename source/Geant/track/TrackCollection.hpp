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
 * @brief Collection of Track state.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackState.hpp"
#include <cstdint>
#include <vector>

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
  const value_type &Get(TrackId_t i) const { return fTracks.at(i); }
  value_type &Get(TrackId_t i) { return fTracks.at(i); }

  template <typename PT>
  friend class TrackModifier;
  friend class TrackAccessor;
};

} // namespace geantx
