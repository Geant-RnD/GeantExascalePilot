#pragma once

#include "Geant/track/Types.hpp"

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
struct TrackHistoryState {
  int fEvent             = 0;  /** Event number */
  int fPrimaryIndx       = 0;  /** Index of the primary particle in the current event */
  ParticleId_t fParticle = 0;  /** Unique particle ID within the event */
  ParticleId_t fMother   = 0;  /** ID of mother particle within the event*/
  int fNsteps            = 0;  /** Number of steps made in this track */
  int fGeneration        = 0;  /** Number of generations to initial particle (0 for primary) */
};

struct TrackGeometryState {
  // TODO: fVolume is a cached 'fPath->Top()->GetLogicalVolume()'
  Volume_t const *fVolume = nullptr; /** Current volume the particle is in */
  VolumePath_t *fPath     = nullptr; /** Current volume state */
  VolumePath_t *fNextpath = nullptr; /** Next volume state */
  double fSnext           = 0;       /** Straight distance to next boundary */
  double fSafety          = 0;       /** Safe distance to any boundary */
  int fMaxDepth           = 0;       /** Maximum geometry depth */
  bool fIsOnBoundaryPreStp = false;  /** Particle was on boundary at the pre-step point */
  bool fBoundary          = false;   /** Starting from boundary */
  bool fPending           = false;   /** Track pending to be processed */
};

struct TrackMaterialState {
  MaterialId_t fMaterial = 0;
};

//---------------------------------------------------------------------------//
/*!
 * \brief Individual physics process state
 *
 * Each physics *process* tracks the number of mean free paths to the next
 * interaction/collision.
 */
struct PhysicsProcessState {
    double fNumOfInteractLengthLeft;
    double fPhysicsInteractLength;
    /* any other state data ??? */
};

constexpr size_t kNumPhysicsProcess = 10;

//---------------------------------------------------------------------------//
/*!
 * \brief State information about the track's physics.
 *
 * TODO: make the physics process a pointer (or type-safe)
 * TODO: change fEindex to an enum returned by the physics process
 * TODO: replace with static lookup based on species
 * TODO: replace with velocity to reduce state size?
 */
struct TrackPhysicsState {
#if 1
  // Consider replacing with lookup values or are those used too often?
  Species_t fSpecies = kHadron; /** Particle species */
  double fMass       = 0;       /** Particle mass */
  int fCharge        = 0;       /** Particle charge */
#endif
  ParticleType_t fParticleType = 0;

  double fMomentum   = 0;       /** Momentum */
  double fEnergy     = 0;       /** Energy (including rest mass energy) */
  double fLogEkin    = 0;       /** Logarithm of energy */

  double fEdep  = 0; /** Energy deposition in the step */
  double fPstep = 0; /** Distance before the next physics interaction */
  ProcessId_t fProcess = 0; /** ID of physics 'process' at the next interaction */

  /* Don't use: int fEindex; -1 for continuous, 1000 for discrete */

  // TODO:
  PhysicsProcessState fProcessState[kNumPhysicsProcess];
};

//---------------------------------------------------------------------------//
/*!
 * \brief Track state
 */
struct TrackState {
  TrackStatus_t fStatus = kAlive; /** Track status */
  double fStep          = 0;      /** Step length being travelled */

  Vector3D_t fPos = 0; /** Position */
  Vector3D_t fDir = 0; /** Direction */
  double fTime = 0; /** Time at beginning of step */

  /* don't use: ESimulationStage fStage */
  /* unused: double fintlen = 0; */
  /* unused: double fnintlen = 0; */
  /* unused: bool fownpath = false; */
  /* unused: bool fprepropagationdone = false; */

  /* don't use: TrackSchedulingState fSchedulingState; */
  TrackHistoryState fHistoryState;
  TrackPhysicsState fPhysicsState;
  TrackMaterialState fMaterialState;
  TrackGeometryState fGeometryState;
};

} // namespace geantx