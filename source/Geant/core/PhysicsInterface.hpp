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
 * @brief Physics interface to real physics simulation.
 */
//===----------------------------------------------------------------------===//

// Originated in the GeantV project.

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/Typedefs.hpp"

#include "Geant/core/Fwd.hpp"
#include "VecGeom/base/Global.h"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {
class Propagator;
class SimulationStage;
class TrackDataMgr;
} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx

/**
 * @brief Class describing physics interface
 */
class PhysicsInterface {
public:
  using TaskData     = geantx::TaskData;
  using TrackDataMgr = geantx::TrackDataMgr;

public:
  /**
   * @brief PhysicsInterface constructor
   */
  PhysicsInterface() {}

  /** @brief PhysicsInterface destructor */
  virtual ~PhysicsInterface();

  /** @brief Attach task data if needed */
  virtual void AttachUserData(TaskData *) {}

  /** @brief Function of initialization */
  virtual void Initialize() {}

  // Interface methods to obtain physics realted symulation stages when V3 is used.
  // These methods are called from the geantx::Propagator::CreateSimulationStages
  // methods (when real-physics is used) to obtain the pointers to the physics
  // simulation stages defined in the real-physics library.
  /** @brief Obtain/create physics step limit computation stage.
   *
   * @param[in,out] prop  Pointer to the propagator object that requires the simulation
   * stage.
   * @return     Pointer to a created ComputeIntLen real-physics simulation stage
   * object.
   */
  virtual geantx::SimulationStage *CreateFastSimStage(geantx::Propagator *prop)     = 0;
  virtual geantx::SimulationStage *CreateComputeIntLStage(geantx::Propagator *prop) = 0;
  virtual geantx::SimulationStage *CreatePrePropagationStage(
      geantx::Propagator *prop) = 0;
  virtual geantx::SimulationStage *CreatePostPropagationStage(
      geantx::Propagator *prop) = 0;

  /** @brief Obtain/create along step action (continuous part) computation stage.
   *
   * @param[in,out] prop  Pointer to the propagator object that requires the simulation
   * stage.
   * @return     Pointer to a created AlongStepAction real-physics simulation stage
   * object.
   */
  virtual geantx::SimulationStage *CreateAlongStepActionStage(
      geantx::Propagator *prop) = 0;
  /** @brief Obtain/create post step action (discrete part) computation stage.
   *
   * @param[in,out] prop  Pointer to the propagator object that requires the simulation
   * stage.
   * @return     Pointer to a created PostStepAction real-physics simulation stage
   * object.
   */
  virtual geantx::SimulationStage *CreatePostStepActionStage(
      geantx::Propagator *prop) = 0;

  virtual geantx::SimulationStage *CreateAtRestActionStage(geantx::Propagator *prop) = 0;
};
