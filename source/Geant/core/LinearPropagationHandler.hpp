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
 * @brief Interface between the scheduler and the linear propagation.
 *
 * Originated from GeantV
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/track/TrackAccessor.hpp"
#include "Geant/track/TrackState.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

class TaskData;

/**
 * @brief Handler grouping charged tracks and performing field propagation.
 */

class LinearPropagationHandler {

protected:
private:
  LinearPropagationHandler(const LinearPropagationHandler &) = delete;
  LinearPropagationHandler &operator=(const LinearPropagationHandler &) = delete;

public:
  /** @brief Default constructor */
  LinearPropagationHandler() = default;

  /** @briefdestructor */
  ~LinearPropagationHandler() = default;

  /** @brief Scalar DoIt interface */
  VECCORE_ATT_HOST_DEVICE
  bool Propagate(TrackState &track, TaskData *td);

  VECCORE_ATT_HOST_DEVICE
  bool IsSameLocation(TrackState &track, TaskData *td) const;
};

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
