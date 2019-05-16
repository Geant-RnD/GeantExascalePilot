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

#include "Geant/core/Common.hpp"
#include "Geant/core/Config.hpp"
#include "Geant/core/Macros.hpp"
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

inline namespace cuda {
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

inline void stream_sync(cudaStream_t _stream)
{
  cudaStreamSynchronize(_stream);
  CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

inline void event_sync(cudaEvent_t _event)
{
  cudaEventSynchronize(_event);
  CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

template <typename _Tp>
_Tp *gpu_malloc(uintmax_t size)
{
  _Tp *_gpu;
  CUDA_CHECK_CALL(cudaMalloc(&_gpu, size * sizeof(_Tp)));
  if (_gpu == nullptr) {
    int _device = 0;
    cudaGetDevice(&_device);
    std::stringstream ss;
    ss << "Error allocating memory on GPU " << _device << " of size "
       << (size * sizeof(_Tp)) << " and type " << typeid(_Tp).name()
       << " (type size = " << sizeof(_Tp) << ")";
    throw std::runtime_error(ss.str().c_str());
  }
  return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void cpu2gpu_memcpy(_Tp *_gpu, const _Tp *_cpu, uintmax_t size, cudaStream_t stream)
{
  cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
  CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu2cpu_memcpy(_Tp *_cpu, const _Tp *_gpu, uintmax_t size, cudaStream_t stream)
{
  cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
  CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu2gpu_memcpy(_Tp *_dst, const _Tp *_src, uintmax_t size, cudaStream_t stream)
{
  cudaMemcpyAsync(_dst, _src, size * sizeof(_Tp), cudaMemcpyDeviceToDevice, stream);
  CUDA_CHECK_LAST_ERROR();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu_memset(_Tp *_gpu, int value, uintmax_t size, cudaStream_t stream)
{
  cudaMemsetAsync(_gpu, value, size * sizeof(_Tp), stream);
  CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

template <typename _Tp>
_Tp *gpu_malloc_and_memcpy(const _Tp *_cpu, uintmax_t size, cudaStream_t stream)
{
  _Tp *_gpu = gpu_malloc<_Tp>(size);
  cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
  CUDA_CHECK_LAST_ERROR();
  return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp *gpu_malloc_and_memset(uintmax_t size, int value, cudaStream_t stream)
{
  _Tp *_gpu = gpu_malloc<_Tp>(size);
  cudaMemsetAsync(_gpu, value, size * sizeof(_Tp), stream);
  CUDA_CHECK_LAST_ERROR();
  return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu2cpu_memcpy_and_free(_Tp *_cpu, _Tp *_gpu, uintmax_t size, cudaStream_t stream)
{
  cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
  CUDA_CHECK_LAST_ERROR();
  cudaFree(_gpu);
  CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

inline cudaStream_t *create_streams(const int nstreams,
                                    unsigned int flag = cudaStreamDefault)
{
  cudaStream_t *streams = new cudaStream_t[nstreams];
  for (int i = 0; i < nstreams; ++i) {
    cudaStreamCreateWithFlags(&streams[i], flag);
    CUDA_CHECK_LAST_ERROR();
  }
  return streams;
}

//======================================================================================//

inline void destroy_streams(cudaStream_t *streams, const int nstreams)
{
  for (int i = 0; i < nstreams; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    CUDA_CHECK_LAST_ERROR();
  }
  delete[] streams;
}

//======================================================================================//

} // namespace cuda

} // namespace geantx
