//===------------------ Geant-V ---------------------------------*- C++ -*-===//
//
//                     Geant-V Prototype
//
//===----------------------------------------------------------------------===//
/**
 * @file GeantCudaUtils.h
 * @brief Utility routines for NVCC/CUDA compilation and setup
 * @author Philippe Canal
 */
//===----------------------------------------------------------------------===//

#pragma once

#ifndef GEANT_CONFIG_H
#include "Geant/core/Config.hpp"
#endif

#ifndef GEANT_ERROR_H
#include "Geant/core/Error.hpp"
#endif

#include <cuda.h>
// #include "driver_types.h" // Required for cudaError_t type
#include "cuda_runtime.h" // Required for cudaGetErrorString

#define GEANT_CUDA_ERROR(err) (geantx::HandleCudaError(err, __FILE__, __LINE__))

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {
inline void HandleCudaError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess) {
    ::geantx::Fatal("Cuda", "%s (%d) in %s at line %d\n", cudaGetErrorString(err), err,
                    file, line);
  }
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
