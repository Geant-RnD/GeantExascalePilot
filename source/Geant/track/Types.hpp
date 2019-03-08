
namespace geantx {

using Vector3        = vecgeom::Vector3D<double>;
using ParticleId_t   = unsigned int;
using ParticleType_t = unsigned int;
using VolumeId_t     = unsigned int;
using MaterialId_t   = unsigned int;
using ProcessId_t    = unsigned int;

//! Physics
enum Species_t { kHadron, kLepton };

//! Track
enum TrackStatus_t { kAlive, kKilled, kInFlight, kBoundary, kExitingSetup, kPhysics, kPostponed, kNew };

/** Basket simulation stages. */
enum ESimulationStage {
  kBeginStage,        // Actions at the beginning of the step
  kComputeIntLStage,    // Physics interaction length computation stage
  kGeometryStepStage,   // Compute geometry transport length
  kPrePropagationStage, // Special msc stage for step limit phase
  /*  kGeometryStepStage,        // Compute geometry transport length */
  kPropagationStage,     // Propagation in field stage
  kPostPropagationStage, // Special msc stage for along-step action stage
  /*  kMSCStage,               // Multiple scattering stage */
  kAlongStepActionStage, // Along step action stage (continuous part of the interaction)
  kPostStepActionStage,  // Post step action stage (discrete part of the interaction)
  kAtRestActionStage,    // At-rest action stage (at-rest part of the interaction)
  kSteppingActionsStage  // User actions
};

/* TransportAction_t is unused */

} // namespace geantx
