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
#include "Geant/core/CudaDeviceInfo.hpp"

#include <PTL/Utility.hh>
#include <PTL/AutoLock.hh>

#include <string>
#include <deque>

#include <cuda.h>
#include <cuda_runtime_api.h>

//======================================================================================//

// Macros to test the cuda error code and issue an error message and throw an
// exception.
//
// GEANT_CUDA_CALL(cond) : use to check the return value of a cuda function.
// GEANT_CUDA_CALL_FUNCTION(cond) : same as GEANT_CUDA_CALL including printing the name
//                                  of the current function.
// GEANT_CUDA_CHECK_LAST_ERROR() : check the 'last' error record by cuda.
// GEANT_CUDA_CHECK_LAST_ERROR() : check the 'last' error record by cuda including
//                                 printing the name of the current function.
// GEANT_CUDA_CHECK_LAST_ERROR_SYNC : synchronize cuda before calling
//                                    GEANT_CUDA_CHECK_LAST_ERROR
// GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC : synchronize cuda before calling
//                                             GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION

#define GEANT_CUDA_CHECK_LAST_ERROR_SYNC() \
  {                                        \
    cudaStreamSynchronize(0);              \
    GEANT_CUDA_CHECK_LAST_ERROR();         \
  }

#define GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC() \
  {                                                 \
    cudaStreamSynchronize(0);                       \
    GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION();         \
  }

#define GEANT_CUDA_CHECK_LAST_ERROR()                                                   \
  {                                                                                     \
    cudaError_t result = cudaGetLastError();                                            \
    if (GEANT_UNLIKELY(result != cudaSuccess)) {                                        \
      cudaGetLastError();                                                               \
      ::geantx::cudaruntime::CudaError(cudaGetErrorString(result), __FILE__, __LINE__); \
    }                                                                                   \
  }

#define GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION()                                       \
  {                                                                                  \
    cudaError_t result = cudaGetLastError();                                         \
    if (GEANT_UNLIKELY(result != cudaSuccess)) {                                     \
      cudaGetLastError();                                                            \
      ::geantx::cudaruntime::CudaErrorFunc(cudaGetErrorString(result), __FUNCTION__, \
                                           __FILE__, __LINE__);                      \
    }                                                                                \
  }

#define GEANT_CUDA_CALL(cond)                                                       \
  {                                                                                 \
    cudaError_t result = (cond);                                                    \
    if (GEANT_UNLIKELY(result != cudaSuccess)) {                                    \
      cudaGetLastError();                                                           \
      ::geantx::cudaruntime::CudaError(cudaGetErrorString(result), #cond, __FILE__, \
                                       __LINE__);                                   \
    }                                                                               \
  }

#define GEANT_CUDA_CALL_FUNCTION(cond)                                        \
  {                                                                           \
    cudaError_t result = (cond);                                              \
    if (GEANT_UNLIKELY(result != cudaSuccess)) {                              \
      cudaGetLastError();                                                     \
      ::geantx::cudaruntime::CudaErrorFunc(cudaGetErrorString(result), #cond, \
                                           __FUNCTION__, __FILE__, __LINE__); \
    }                                                                         \
  }

namespace geantx {
inline namespace cudaruntime {

//======================================================================================//

void CudaError(const char *err_string, const char *msg, const char *file, int line);

void CudaErrorFunc(const char *err_string, const char *msg, const char *function,
                   const char *file, int line);

void CudaError(const char *err_string, const char *file, int line);

void CudaErrorFunc(const char *err_string, const char *function, const char *file,
                   int line);

//======================================================================================//

inline void stream_sync(cudaStream_t _stream)
{
  cudaStreamSynchronize(_stream);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION();
}

//======================================================================================//

inline void event_sync(cudaEvent_t _event)
{
  cudaEventSynchronize(_event);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION();
}

//======================================================================================//

template <typename _Tp>
_Tp *gpu_malloc(uintmax_t size)
{
  _Tp *_gpu;
  GEANT_CUDA_CALL(cudaMalloc(&_gpu, size * sizeof(_Tp)));
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
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu2cpu_memcpy(_Tp *_cpu, const _Tp *_gpu, uintmax_t size, cudaStream_t stream)
{
  cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu2gpu_memcpy(_Tp *_dst, const _Tp *_src, uintmax_t size, cudaStream_t stream)
{
  cudaMemcpyAsync(_dst, _src, size * sizeof(_Tp), cudaMemcpyDeviceToDevice, stream);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu_memset(_Tp *_gpu, int value, uintmax_t size, cudaStream_t stream)
{
  cudaMemsetAsync(_gpu, value, size * sizeof(_Tp), stream);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
}

//======================================================================================//

template <typename _Tp>
_Tp *gpu_malloc_and_memcpy(const _Tp *_cpu, uintmax_t size, cudaStream_t stream)
{
  _Tp *_gpu = gpu_malloc<_Tp>(size);
  cudaMemcpyAsync(_gpu, _cpu, size * sizeof(_Tp), cudaMemcpyHostToDevice, stream);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
  return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp *gpu_malloc_and_memset(uintmax_t size, int value, cudaStream_t stream)
{
  _Tp *_gpu = gpu_malloc<_Tp>(size);
  cudaMemsetAsync(_gpu, value, size * sizeof(_Tp), stream);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
  return _gpu;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void gpu2cpu_memcpy_and_free(_Tp *_cpu, _Tp *_gpu, uintmax_t size, cudaStream_t stream)
{
  cudaMemcpyAsync(_cpu, _gpu, size * sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
  cudaFree(_gpu);
  GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
}

//======================================================================================//

inline cudaStream_t *create_streams(const int nstreams,
                                    unsigned int flag = cudaStreamDefault)
{
  cudaStream_t *streams = new cudaStream_t[nstreams];
  for (int i = 0; i < nstreams; ++i) {
    cudaStreamCreateWithFlags(&streams[i], flag);
    GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
  }
  return streams;
}

//======================================================================================//

inline void destroy_streams(cudaStream_t *streams, const int nstreams)
{
  for (int i = 0; i < nstreams; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    GEANT_CUDA_CHECK_LAST_ERROR_FUNCTION_SYNC();
  }
  delete[] streams;
}

//======================================================================================//

template <typename _Func, typename... _Args>
void RunAlgorithm(_Func cpu_func, _Func cuda_func, _Args... args)
{
  using PTL::GetEnv;
  bool use_cpu = GetEnv<bool>("GEANT_USE_CPU", false);
  if (use_cpu) {
    try {
      cpu_func(std::forward<_Args>(args)...);
    } catch (const std::exception &e) {
      PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
      std::cerr << e.what() << '\n';
    }
    return;
  }

  std::deque<DeviceOption> options;
  options.push_back(DeviceOption(0, "cpu", "Run on CPU"));
  options.push_back(DeviceOption(1, "gpu", "Run on GPU with CUDA"));

  std::string default_key = "gpu";

  auto default_itr =
      std::find_if(options.begin(), options.end(),
                   [&](const DeviceOption &itr) { return (itr == default_key); });

  //------------------------------------------------------------------------//
  auto print_options = [&]() {
    static bool first = true;
    if (!first)
      return;
    else
      first = false;

    std::stringstream ss;
    DeviceOption::Header(ss);
    for (const auto &itr : options) {
      ss << itr;
      if (itr == *default_itr) ss << "\t(default)";
      ss << "\n";
    }
    DeviceOption::Footer(ss);

    PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
    std::cout << "\n" << ss.str() << std::endl;
  };
  //------------------------------------------------------------------------//
  auto print_selection = [&](DeviceOption &selected_opt) {
    static bool first = true;
    if (!first)
      return;
    else
      first = false;

    std::stringstream ss;
    DeviceOption::Spacer(ss, '-');
    ss << "Selected device: " << selected_opt << "\n";
    DeviceOption::Spacer(ss, '-');

    PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
    std::cout << ss.str() << std::endl;
  };
  //------------------------------------------------------------------------//

  // Run on CPU if nothing available
  if (options.size() <= 1) {
    cpu_func(std::forward<_Args>(args)...);
    return;
  }

  // print the GPU execution type options
  print_options();

  default_key = default_itr->key;
  auto key    = GetEnv("GEANT_DEVICE", default_key);

  auto selection = std::find_if(options.begin(), options.end(),
                                [&](const DeviceOption &itr) { return (itr == key); });

  if (selection == options.end())
    selection = std::find_if(options.begin(), options.end(),
                             [&](const DeviceOption &itr) { return itr == default_key; });

  print_selection(*selection);

  try {
    switch (selection->index) {
    case 0:
      cpu_func(std::forward<_Args>(args)...);
      break;
    case 1:
      cuda_func(std::forward<_Args>(args)...);
      break;
    default:
      cpu_func(std::forward<_Args>(args)...);
      break;
    }
  } catch (std::exception &e) {
    if (selection != options.end() && selection->index != 0) {
      {
        PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
        std::cerr << "[TID: " << GetThisThreadID() << "] " << e.what() << std::endl;
        std::cerr << "[TID: " << GetThisThreadID() << "] "
                  << "Falling back to CPU algorithm..." << std::endl;
      }
      try {
        cpu_func(std::forward<_Args>(args)...);
      } catch (std::exception &_e) {
        std::stringstream ss;
        ss << "\n\nError executing :: " << _e.what() << "\n\n";
        {
          PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
          std::cerr << _e.what() << std::endl;
        }
        throw std::runtime_error(ss.str().c_str());
      }
    }
  }
}

//======================================================================================//

} // namespace cudaruntime

} // namespace geantx
