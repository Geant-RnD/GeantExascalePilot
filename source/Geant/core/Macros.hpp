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
 * @brief Utility macros
 */
//===----------------------------------------------------------------------===//

#pragma once

//======================================================================================//
//  C headers
//
#include <cassert>
#include <cstdint>
#include <cstdlib>

//======================================================================================//
//  C++ headers
//
#include <type_traits>
#include <chrono>
#include <sstream>

//--------------------------------------------------------------------------------------//
// cuda headers for debug macros
//
#include <cuda.h>
#include <cuda_runtime_api.h>

//======================================================================================//

#if !defined(CUDA_CHECK_LAST_ERROR)
#define CUDA_CHECK_LAST_ERROR()                                               \
  {                                                                           \
    cudaStreamSynchronize(0);                                                 \
    cudaError err = cudaGetLastError();                                       \
    if (cudaSuccess != err) {                                                 \
      std::stringstream ss;                                                   \
      ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__ \
         << "':" << __LINE__ << " : " << cudaGetErrorString(err);             \
      fprintf(stderr, "%s\n", ss.str().c_str());                              \
      throw std::runtime_error(ss.str().c_str());                             \
    }                                                                         \
  }
#endif

//======================================================================================//

#if !defined(CUDA_DEBUG_LAST_ERROR)
#if defined(DEBUG)
#define CUDA_DEBUG_LAST_ERROR()                                               \
  {                                                                           \
    cudaStreamSynchronize(0);                                                 \
    cudaError err = cudaGetLastError();                                       \
    if (cudaSuccess != err) {                                                 \
      std::stringstream ss;                                                   \
      ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__ \
         << "':" << __LINE__ << " : " << cudaGetErrorString(err);             \
      fprintf(stderr, "%s\n", ss.str().c_str());                              \
      throw std::runtime_error(ss.str().c_str());                             \
    }                                                                         \
  }
#else
#define CUDA_DEBUG_LAST_ERROR() \
  {                             \
    ;                           \
  }
#endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(CUDA_CHECK_CALL)
#define CUDA_CHECK_CALL(err)                                                  \
  {                                                                           \
    if (cudaSuccess != err) {                                                 \
      std::stringstream ss;                                                   \
      ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__ \
         << "':" << __LINE__ << " : " << cudaGetErrorString(err);             \
      fprintf(stderr, "%s\n", ss.str().c_str());                              \
      throw std::runtime_error(ss.str().c_str());                             \
    }                                                                         \
  }
#endif


