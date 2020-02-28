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

#include "Geant/core/Typedefs.hpp"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"

namespace geantx
{
using VolumePath_t    = VECGEOM_NAMESPACE::NavigationState;
using Volume_t        = VECGEOM_NAMESPACE::LogicalVolume;
using ThreeVector     = VECGEOM_NAMESPACE::Vector3D<double>;
using Material_t      = geantx::Material;
using ParticleId_t    = unsigned int;
using ParticleDefId_t = unsigned int;
using VolumeId_t      = unsigned int;
using MaterialId_t    = unsigned int;
using ProcessId_t     = unsigned int;

//! Physics
enum class ParticleSpecies : short
{
    Hadron,
    Lepton
};

//! Track
enum class TrackStatus : short
{
    Alive,
    Killed,
    InFlight,
    Boundary,
    ExitingSetup,
    Physics,
    Postponed,
    New
};

/** Basket simulation stages. */
enum class SimulationStage : short
{
    BeginStage,           // Actions at the beginning of the step
    ComputeIntLStage,     // Physics interaction length computation stage
    GeometryStepStage,    // Compute geometry transport length
    PrePropagationStage,  // Special msc stage for step limit phase
    // GeometryStepStage,     // Compute geometry transport length
    PropagationStage,      // Propagation in field stage
    PostPropagationStage,  // Special msc stage for along-step action stage
    // MSCStage,              // Multiple scattering stage
    AlongStepActionStage,  // Along step action stage (continuous part of the interaction)
    PostStepActionStage,   // Post step action stage (discrete part of the interaction)
    AtRestActionStage,     // At-rest action stage (at-rest part of the interaction)
    SteppingActionsStage   // User actions
};

/* TransportAction_t is unused */

}  // namespace geantx

#include <ostream>

namespace std
{
inline std::ostream&
operator<<(std::ostream& os, const geantx::TrackStatus& status)
{
    using geantx::TrackStatus;
    switch(status)
    {
        case TrackStatus::Alive: os << "Alive"; break;
        case TrackStatus::Killed: os << "Killed"; break;
        case TrackStatus::InFlight: os << "InFlight"; break;
        case TrackStatus::Boundary: os << "Boundary"; break;
        case TrackStatus::ExitingSetup: os << "ExitingSetup"; break;
        case TrackStatus::Physics: os << "Physics"; break;
        case TrackStatus::Postponed: os << "Postponed"; break;
        case TrackStatus::New: os << "New"; break;
        default: os << "Unknown"; break;
    }
    return os;
}
}  // namespace std
