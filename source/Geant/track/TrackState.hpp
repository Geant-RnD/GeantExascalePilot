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
 * @brief Track information split in related categories.  See also the Track
 *        Accessors and modifiers.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/Types.hpp"
#include "timemory/timemory.hpp"

#include <sstream>

namespace geantx {

#if 0
//---------------------------------------------------------------------------//
/*!
 * \brief Internals for scheduling.
 */
struct TrackSchedulingState {
  int fGVcode = 0; /** GV particle code */
  int fBindex = 0; /** Index in the track block */
  int fEvslot = 0; /** Event slot */
};
#endif

//---------------------------------------------------------------------------//
/*!
 * \brief Counters for the current particle track.
 *
 * An \em event is a collection of *correlated* \em primary particle tracks
 * emanating from a single point in space. This could represent, for example,
 * the decay of a hypothetized particle.
 *
 * Each \em primary particle can create secondary particles. The state of each
 * existing particle is a \em track . Each movement/time step the particle
 * takes is a \em step and increments the step counter.
 *
 * Each unique particle created by the event (this *does* span primary
 * particles) has a unique particle ID. A secondary particle saves the
 * \em mother particle's ID and increments the \em generation counter by
 * one.
 */
struct TrackHistoryState
{
    int fEvent             = 0; /** Event number */
    int fPrimaryIndx       = 0; /** Index of the primary particle in the current event */
    ParticleId_t fParticle = 0; /** Unique particle ID within the event */
    ParticleId_t fMother   = 0; /** ID of mother particle within the event*/
    int          fNsteps   = 0; /** Number of steps made in this track */
    int          fGeneration = 0; /** Num of generations since creation (primary==0) */
};

struct TrackGeometryState
{
    // TODO: fVolume is a cached 'fPath->Top()->GetLogicalVolume()'
    Volume_t const* fVolume   = nullptr; /** Current volume the particle is in */
    VolumePath_t*   fPath     = nullptr; /** Current volume state */
    VolumePath_t*   fNextpath = nullptr; /** Next volume state */
    double          fSnext    = 0;       /** Straight distance to next boundary */
    double          fSafety   = 0;       /** Safe distance to any boundary */
    int             fMaxDepth = 0;       /** Maximum geometry depth */
    bool            fIsOnBoundaryPreStp =
        false;              /** Particle was on boundary at the pre-step point */
    bool fBoundary = false; /** Starting from boundary */
    bool fPending  = false; /** Track pending to be processed  (???) */
};

struct TrackMaterialState
{
  MaterialId_t fMaterialId = 0;
  Material_t*  fMaterial = nullptr; /** ptr to the current Material */
  /* replace Material_t* by MaterialCut*  */
};

//---------------------------------------------------------------------------//
/*!
 * \brief Individual physics process state
 *
 * Each physics *process* tracks the number of mean free paths to the next
 * interaction/collision.
 */
struct PhysicsProcessState
{
    double fNumOfInteractLengthLeft;
    double fPhysicsInteractLength;
    /* any other state data ??? */
};

// DEPRECATED: J. Madsen (10/18/19)
//      - handling this elsewhere
// constexpr size_t kNumPhysicsProcess = 10;

//---------------------------------------------------------------------------//
/*!
 * \brief State information about the track's physics.
 *
 * TODO: make the physics process a pointer (or type-safe)
 * TODO: change fEindex to an enum returned by the physics process
 */
struct TrackPhysicsState
{
    ParticleDefId_t fParticleDefId = 0; /** Index into possible particle definitions */

    double fEkin     = 0; /** Kinetic energy */
    double fMomentum = 0; /** Relativistic momentum */

    double      fEdep    = 0; /** Energy deposition in the step */
    double      fPstep   = 0; /** Distance before the next physics interaction */
    ProcessId_t fProcess = 0; /** ID of physics 'process' at the next interaction */

    /* Don't use: int fEindex; -1 for continuous, 1000 for discrete */

    // DEPRECATED: J. Madsen (10/18/19)
    //      - handling this elsewhere
    // PhysicsProcessState fProcessState[kNumPhysicsProcess];
};

//---------------------------------------------------------------------------//
/*!
 * \brief Track state
 */
struct TrackState
{
    TrackState()                  = default;
    ~TrackState()                 = default;
    TrackState(const TrackState&) = default;
    TrackState(TrackState&&)      = default;
    TrackState& operator=(const TrackState&) = default;
    TrackState& operator=(TrackState&&) = default;

    TrackStatus fStatus = TrackStatus::Alive; /** Track status */
    double        fStep   = 0.0;                  /** Step length being travelled */
    double        fTime   = 0.0;                  /** Time at beginning of step */

    ThreeVector fPos = { 0.0, 0.0, 0.0 }; /** Position */
    ThreeVector fDir = { 0.0, 0.0, 0.0 }; /** Direction */

    /* don't use: ESimulationStage fStage */
    /* unused: double fintlen = 0; */
    /* unused: double fnintlen = 0; */
    /* unused: bool fownpath = false; */
    /* unused: bool fprepropagationdone = false; */

    /* don't use: TrackSchedulingState fSchedulingState; */
    TrackHistoryState  fHistoryState;
    TrackPhysicsState  fPhysicsState;
    PhysicsProcessState  fPhysicsProcessState;
    TrackMaterialState fMaterialState;
    TrackGeometryState fGeometryState;

    friend std::ostream& operator<<(std::ostream& os, const TrackState& t)
    {
        // the tuple<string> overload of tim::apply changes the definition of join
        // to join the entries with second separator and then join the paired
        // entries with the first separator
        using apply_t = tim::apply<std::tuple<std::string>>;
        auto&& labels = std::make_tuple("addr", "status", "step", "pos", "dir", "time");
        auto&& values = std::make_tuple(&t, t.fStatus, t.fStep, t.fPos, t.fDir, t.fTime);
        os << apply_t::join(", ", "=", labels, values);
        return os;
    }
};

// Those need to be transfered to the appropriate State.
GEANT_FORCE_INLINE
bool IsAlive(const TrackState &state)
{
    return state.fStatus != TrackStatus::Killed;
}

GEANT_FORCE_INLINE
bool IsStopped(const TrackState &state)
{
    return state.fPhysicsState.fEkin <= 0.0;
}

}  // namespace geantx
