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
 * @brief Utility routines for NVCC/CUDA compilation and setup
 */
//===----------------------------------------------------------------------===//

// Originated in the GeantV project.

#pragma once

#ifndef GEANT_CONFIG_H
#  include "Geant/core/Config.hpp"
#endif

#include "Geant/core/Logger.hpp"

#include <cuda.h>
// #include "driver_types.h" // Required for cudaError_t type
#include "cuda_runtime.h" // Required for cudaGetErrorString

#define GEANT_CUDA_ERROR(err) (geantx::HandleCudaError(err, __FILE__, __LINE__))

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {
inline void HandleCudaError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess) {
    ::geantx::Log(kFatal).From("Cuda") << cudaGetErrorString(err) << "(" << err << ") in "
                                       << file << " at line " << line;
  }
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
