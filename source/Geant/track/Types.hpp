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
 * @brief Declare a few of the generic types needed by GeantX
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <base/Vector3D.h>
#include <navigation/NavigationState.h>
#include <volumes/LogicalVolume.h>
#include "Geant/core/Typedefs.hpp"

namespace geantx {

using VolumePath_t    = VECGEOM_NAMESPACE::NavigationState;
using Volume_t        = VECGEOM_NAMESPACE::LogicalVolume;
using ThreeVector     = VECGEOM_NAMESPACE::Vector3D<double>;
using ParticleId_t    = unsigned int;
using ParticleDefId_t = unsigned int;
using VolumeId_t      = unsigned int;
using MaterialId_t    = unsigned int;
using ProcessId_t     = unsigned int;

//! Physics
enum Species_t { kHadron, kLepton };

//! Track
enum TrackStatus_t {
  kAlive,
  kKilled,
  kInFlight,
  kBoundary,
  kExitingSetup,
  kPhysics,
  kPostponed,
  kNew
};

/** Basket simulation stages. */
enum ESimulationStage {
  kBeginStage,          // Actions at the beginning of the step
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
