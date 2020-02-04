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
 * @brief Accessor to the common/base part of a Track state.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackCollection.hpp"
#include "Geant/track/TrackState.hpp"
#include "Geant/track/Types.hpp"

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
  {}

  // >>> ACCESSORS

  const ThreeVector &Position() const { return this->State().fPos; }
  const ThreeVector &Direction() const { return this->State().fDir; }
  double Step() const { return this->State().fStep; }

  ParticleId_t Id() const { return this->State().fHistoryState.fParticle; }
  TrackStatus Status() const { return this->State().fStatus; }
  bool Alive() const { return this->Status() != TrackStatus::Killed; }

protected:
  // >>> IMPLEMENTATION DETAILS

  explicit TrackAccessor(const TrackState &state) : fState(state) {}
  const TrackState &State() const { return fState; }

  template <typename PT>
  friend class TrackModifier;
};

} // namespace geantx
