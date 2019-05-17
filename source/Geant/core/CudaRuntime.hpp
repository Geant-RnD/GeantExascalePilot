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
#include "Geant/core/Macros.hpp"
#include <PTL/Utility.hh>
#include <libgen.h>
#include <string>

namespace geantx {
inline namespace cudaruntime {

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
