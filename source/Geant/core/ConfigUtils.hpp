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
 * @brief CUDA helper routines.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include <PTL/Utility.hh>
#include <libgen.h>
#include <string>

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {

inline std::string GetDataFileLocation(int argc, char **argv, const char *dataFilename)
{
  const char *dir = argc > 0 ? dirname(argv[0]) : ".";

  if (dir == nullptr) dir = ".";
  return std::string(dir) + "/" + dataFilename;
}

inline std::string GetDataFileLocation(int argc, char **argv, std::string dataFilename)
{
  return GetDataFileLocation(argc, argv, dataFilename.c_str());
}

} // namespace GEANT_IMPL_NAMESPACE

//======================================================================================//

inline namespace cudaruntime {
//======================================================================================//

inline int GetNumMasterStreams(const int &init = 1)
{
  return std::max(PTL::GetEnv<int>("GEANT_NUM_STREAMS", init), 1);
}

//======================================================================================//

inline int GetBlockSize(const int &init = 32)
{
  static thread_local int _instance = PTL::GetEnv<int>("GEANT_BLOCK_SIZE", init);
  return _instance;
}

//======================================================================================//

inline int GetGridSize(const int &init = 0)
{
  // default value of zero == calculated according to block and loop size
  static thread_local int _instance = PTL::GetEnv<int>("GEANT_GRID_SIZE", init);
  return _instance;
}

//======================================================================================//

inline int ComputeGridSize(const int &size, const int &block_size = GetBlockSize())
{
  return (size + block_size - 1) / block_size;
}

//======================================================================================//

inline dim3 GetBlockDims(const dim3 &init = dim3(32, 32, 1))
{
  int _x = PTL::GetEnv<int>("GEANT_BLOCK_SIZE_X", init.x);
  int _y = PTL::GetEnv<int>("GEANT_BLOCK_SIZE_Y", init.y);
  int _z = PTL::GetEnv<int>("GEANT_BLOCK_SIZE_Z", init.z);
  return dim3(_x, _y, _z);
}

//======================================================================================//

inline dim3 GetGridDims(const dim3 &init = dim3(0, 0, 0))
{
  // default value of zero == calculated according to block and loop size
  int _x = PTL::GetEnv<int>("GEANT_GRID_SIZE_X", init.x);
  int _y = PTL::GetEnv<int>("GEANT_GRID_SIZE_Y", init.y);
  int _z = PTL::GetEnv<int>("GEANT_GRID_SIZE_Z", init.z);
  return dim3(_x, _y, _z);
}

//======================================================================================//

inline dim3 ComputeGridDims(const dim3 &dims, const dim3 &blocks = GetBlockDims())
{
  return dim3(ComputeGridSize(dims.x, blocks.x), ComputeGridSize(dims.y, blocks.y),
              ComputeGridSize(dims.z, blocks.z));
}

//======================================================================================//

} // namespace cudaruntime

} // namespace geantx
