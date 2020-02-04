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
 * @brief Memory pool for device and host allocations.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Geant/core/Memory.hpp"
#include "Geant/core/Profiler.hpp"

namespace geantx {
//======================================================================================//
// memory namespace for specifying the type of memory
//
namespace memory {
struct host {};
struct device {};
struct pinned {};
} // namespace memory

//======================================================================================//
// device namespace for specifying traits based on device types
//
namespace device {
struct cpu {};
struct gpu {};
} // namespace device

//======================================================================================//
//  type-trait for designating if the type will be offloaded
//
template <typename _Tp>
struct OffloadMemoryPool : std::false_type {};

//======================================================================================//
//  type-trait for using pinned memory or non-pinned memory
//
template <typename _Tp>
struct OffloadMemoryType {
  // use memory::pinned or memory::host
  using type = memory::pinned;
};

//======================================================================================//
//  gets the size of the page
//
inline auto GetPageSize()
{
  return sysconf(_SC_PAGESIZE);
}

//======================================================================================//
//  container of pointers to memory pages
//
namespace details {
// declaration
template <typename _Tp, typename _Mem>
class MemoryPoolImpl;
template <typename _Tp, bool _Offload>
class MemoryPool;

//======================================================================================//
//  a page of memory on a device
//
template <typename _Tp, typename _Mem>
class MemoryPage {
  // only valid for the _Mem types
  static_assert(std::is_same<_Mem, memory::host>::value ||
                    std::is_same<_Mem, memory::device>::value ||
                    std::is_same<_Mem, memory::pinned>::value,
                "MemoryPage<T, Mode> is only valid for modes: host, device, and pinned");

  // allows access to next()
  friend class MemoryPoolImpl<_Tp, _Mem>;

public:
  template <typename _Up>
  using List_t = std::vector<_Up>;
  template <typename _Key, typename _Mapped>
  using Map_t     = std::unordered_map<_Key, _Mapped>;
  using size_type = typename List_t<void *>::size_type;
  using type      = _Tp;
  using this_type = MemoryPage<_Tp, _Mem>;

public:
  MemoryPage();
  ~MemoryPage();

  MemoryPage(const this_type &) = delete;
  MemoryPage(this_type &&)      = default;

  MemoryPage &operator=(const this_type &) = delete;
  MemoryPage &operator=(this_type &&) = default;

public:
  // allocate at a specific offset
  std::tuple<_Tp *, int64_t> alloc(const int64_t &_offset);
  // mark offset as free
  void free(type *ptr);
  // check if pointer is held by page
  bool owns(type *ptr) const { return (m_associated.find(ptr) != m_associated.end()); }
  // check if page is valid
  bool valid() const { return (m_page != nullptr); }
  // check if any pointers are being used
  bool utilized() const { return (m_used > 0); }
  // get the offset of pointer in page
  size_type offset(type *ptr) const { return m_associated.find(ptr)->second; }
  // register pointer and offset
  void insert(type *ptr, const size_type &idx);
  // remove pointer and offset
  void erase(type *ptr);
  // remove pointer and offset
  void erase(const size_type &idx);

  // allow calling this function only for host code
  template <typename _Up                                                       = _Mem,
            std::enable_if_t<(!std::is_same<_Up, memory::device>::value), int> = 0>
  std::tuple<_Tp *, int64_t> alloc()
  {
    return alloc(next());
  }

private:
  // gets the next available slot available
  int64_t next() const;

private:
  int64_t m_offset          = -1;
  int64_t m_used            = 0;
  int64_t m_allocs_per_page = GetPageSize() / sizeof(type);
  type *m_page              = nullptr;
  List_t<bool> m_available  = List_t<bool>(m_allocs_per_page, true);
  Map_t<type *, size_type> m_associated;

private:
  // type-trait dependent allocation functions
  //
  template <typename _Up                                                    = _Mem,
            std::enable_if_t<(std::is_same<_Up, memory::host>::value), int> = 0>
  static type *_alloc(size_type nobjs)
  {
    auto sz = nobjs * sizeof(_Tp);
    return (sz > 0) ? static_cast<type *>(std::malloc(sz)) : nullptr;
  }

  template <typename _Up                                                      = _Mem,
            std::enable_if_t<(std::is_same<_Up, memory::pinned>::value), int> = 0>
  static type *_alloc(size_type nobjs)
  {
    auto sz     = nobjs * sizeof(_Tp);
    void *_page = nullptr;
    if (sz > 0) cudaMallocHost(&_page, sz);
    return static_cast<type *>(_page);
  }

  template <typename _Up                                                      = _Mem,
            std::enable_if_t<(std::is_same<_Up, memory::device>::value), int> = 0>
  static type *_alloc(size_type nobjs)
  {
    auto sz     = nobjs * sizeof(_Tp);
    void *_page = nullptr;
    if (sz > 0) cudaMalloc(&_page, sz);
    return static_cast<type *>(_page);
  }

private:
  // type-trait dependent deallocation functions
  //
  template <typename _Up                                                    = _Mem,
            std::enable_if_t<(std::is_same<_Up, memory::host>::value), int> = 0>
  static void _free(void *_page)
  {
    std::free(_page);
  }

  template <typename _Up                                                      = _Mem,
            std::enable_if_t<(std::is_same<_Up, memory::pinned>::value), int> = 0>
  static void _free(void *_page)
  {
    cudaMallocHost(&_page, GetPageSize());
  }

  template <typename _Up                                                      = _Mem,
            std::enable_if_t<(std::is_same<_Up, memory::device>::value), int> = 0>
  static void _free(void *_page)
  {
    cudaFree(_page);
  }
};

//======================================================================================//
//  a collection of memory pages on a device
//
template <typename _Tp, typename _Mem>
class MemoryPoolImpl {
  // only valid for the _Mem types
  static_assert(
      std::is_same<_Mem, memory::host>::value ||
          std::is_same<_Mem, memory::device>::value ||
          std::is_same<_Mem, memory::pinned>::value,
      "MemoryPoolImpl<T, Mode> is only valid for modes: host, device, and pinned");

public:
  template <typename _Up>
  using List_t = std::vector<_Up>;
  template <typename _Key, typename _Mapped>
  using Map_t      = std::unordered_map<_Key, _Mapped>;
  using type       = _Tp;
  using page_type  = MemoryPage<_Tp, _Mem>;
  using this_type  = MemoryPoolImpl<_Tp, _Mem>;
  using list_type  = List_t<page_type *>;
  using size_type  = typename list_type::size_type;
  using map_type   = Map_t<type *, size_type>;
  using tuple_type = std::tuple<type *, int64_t, int64_t>;

public:
  MemoryPoolImpl(size_type npages = 1);
  ~MemoryPoolImpl();

  MemoryPoolImpl(const this_type &) = delete;
  MemoryPoolImpl(this_type &&)      = default;

  MemoryPoolImpl &operator=(const this_type &) = delete;
  MemoryPoolImpl &operator=(this_type &&) = default;

  tuple_type alloc();
  tuple_type alloc(const tuple_type &);
  void free(type *ptr);

  bool add_page();
  void delete_page();

  size_type size() const { return m_page_list.size(); }

private:
  size_type m_last_page = 0;
  list_type m_page_list;
  map_type m_associated;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, bool _Offload>
class MemoryPool {};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
class MemoryPool<_Tp, false> {
public:
  using CpuPool_t = MemoryPoolImpl<_Tp, memory::host>;
  using GpuPool_t = CpuPool_t;
  using this_type = MemoryPool<_Tp, false>;
  using size_type = std::size_t;

public:
  MemoryPool(size_type npages = 1) : m_cpu(npages) { GEANT_THIS_TYPE_TESTING_MARKER(""); }

  _Tp *alloc()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return std::get<0>(m_cpu.alloc());
  }

  void free(_Tp *&ptr)
  {
    // GEANT_THIS_TYPE_TESTING_MARKER("");
    m_cpu.free(ptr);
    ptr = nullptr;
  }
  bool add_page()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return m_cpu.add_page();
  }
  void delete_page()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    m_cpu.delete_page();
  }

  size_type size() const { return m_cpu.size(); }

  _Tp *get(_Tp *_cpu) const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return _cpu;
  }

  template <typename _Target,
            std::enable_if_t<(std::is_same<_Target, device::gpu>::value), int> = 0>
  void transfer_to(_Target &&, cudaStream_t = 0)
  {}

  template <typename _Target,
            std::enable_if_t<(std::is_same<_Target, device::cpu>::value), int> = 0>
  void transfer_to(_Target &&, cudaStream_t = 0)
  {}

  void transfer(const cudaMemcpyKind &, cudaStream_t = 0) {}

private:
  CpuPool_t m_cpu;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp>
class MemoryPool<_Tp, true> {
public:
  template <typename _Key, typename _Mapped>
  using Map_t     = std::unordered_map<_Key, _Mapped>;
  using Memory_t  = typename OffloadMemoryType<_Tp>::type;
  using CpuPool_t = MemoryPoolImpl<_Tp, Memory_t>;
  using GpuPool_t = MemoryPoolImpl<_Tp, memory::device>;
  using this_type = MemoryPool<_Tp, true>;
  using size_type = std::size_t;

  static_assert(std::is_same<Memory_t, memory::host>::value ||
                    std::is_same<Memory_t, memory::pinned>::value,
                "OffloadMemoryType is only valid for modes: host and pinned");

public:
  MemoryPool(size_type npages = 1) : m_cpu(npages), m_gpu(npages)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
  }

  _Tp *alloc()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _cpu                       = m_cpu.alloc();
    auto _gpu                       = m_gpu.alloc(_cpu);
    m_associated[std::get<0>(_cpu)] = std::get<0>(_gpu);
    return std::get<0>(_cpu);
  }

  template <typename _Target,
            std::enable_if_t<(std::is_same<_Target, device::cpu>::value), int> = 0>
  _Tp *alloc()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _cpu                       = m_cpu.alloc();
    auto _gpu                       = m_gpu.alloc(_cpu);
    m_associated[std::get<0>(_cpu)] = std::get<0>(_gpu);
    return std::get<0>(_cpu);
  }

  template <typename _Target,
            std::enable_if_t<(std::is_same<_Target, device::gpu>::value), int> = 0>
  _Tp *alloc()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _cpu                       = m_cpu.alloc();
    auto _gpu                       = m_gpu.alloc(_cpu);
    m_associated[std::get<0>(_cpu)] = std::get<0>(_gpu);
    return std::get<0>(_gpu);
  }

  void free(_Tp *&ptr)
  {
    // GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _gpu_itr = m_associated.find(ptr);
    auto _gpu     = (_gpu_itr != m_associated.end()) ? _gpu_itr->second : nullptr;
    m_cpu.free(ptr);
    if (_gpu) m_gpu.free(_gpu);
    ptr = nullptr;
  }

  bool add_page()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    auto _cpu = m_cpu.add_page();
    auto _gpu = m_gpu.add_page();
    return (_cpu && _gpu);
  }

  void delete_page()
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    m_cpu.delete_page();
    m_gpu.delete_page();
  }

  size_type size() const { return m_cpu.size(); }

  _Tp *get(_Tp *_cpu) const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return (m_associated.find(_cpu) != m_associated.end())
               ? m_associated.find(_cpu)->second
               : nullptr;
  }

  template <typename _Target,
            std::enable_if_t<(std::is_same<_Target, device::gpu>::value), int> = 0>
  void transfer_to(_Target &&, cudaStream_t stream = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    for (auto &itr : m_associated)
      cudaMemcpyAsync(itr.second, itr.first, sizeof(_Tp), cudaMemcpyHostToDevice, stream);
  }

  template <typename _Target,
            std::enable_if_t<(std::is_same<_Target, device::cpu>::value), int> = 0>
  void transfer_to(_Target &&, cudaStream_t stream = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    for (auto &itr : m_associated)
      cudaMemcpyAsync(itr.first, itr.second, sizeof(_Tp), cudaMemcpyDeviceToHost, stream);
  }

  void transfer(const cudaMemcpyKind &kind, cudaStream_t stream = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (kind == cudaMemcpyDeviceToHost) {
      transfer_to(device::cpu(), stream);
    } else if (kind == cudaMemcpyDeviceToHost) {
      transfer_to(device::gpu(), stream);
    }
  }

private:
  CpuPool_t m_cpu;
  GpuPool_t m_gpu;
  Map_t<_Tp *, _Tp *> m_associated;
};

} // namespace details

//======================================================================================//
//  container of pointers to memory pages
//
template <typename _Tp, bool Offload = OffloadMemoryPool<_Tp>::value>
using MemoryPool = details::MemoryPool<_Tp, Offload>;

//======================================================================================//
//  An allocator that can be inherit from
//
template <typename _Tp, bool Offload = OffloadMemoryPool<_Tp>::value>
struct MemoryPoolAllocator {
  using pool_type = geantx::MemoryPool<_Tp, Offload>;
  using this_type = MemoryPoolAllocator<_Tp, Offload>;

  void *operator new(std::size_t)
  {
    return static_cast<void *>(get_allocator()->alloc());
  }

  void operator delete(void *ptr)
  {
    auto tptr = static_cast<_Tp *>(ptr);
    if (get_allocator()) get_allocator()->free(tptr);
  }

  DevicePtr<_Tp> device_ptr() const
  {
    auto ptr = static_cast<_Tp *>(const_cast<this_type *>(this));
    return DevicePtr<_Tp>(get_allocator()->get(ptr));
  }

  template <typename _Target>
  void transfer_to(_Target &&target, cudaStream_t stream = 0)
  {
    get_allocator()->transfer_to(std::forward<_Target>(target), stream);
  }

  void transfer(const cudaMemcpyKind &kind, cudaStream_t stream = 0)
  {
    get_allocator()->transfer(kind, stream);
  }

private:
  using Alloc_t        = pool_type;
  using AllocPointer_t = std::unique_ptr<Alloc_t>;
  static AllocPointer_t &get_allocator()
  {
    static thread_local AllocPointer_t _instance(new Alloc_t);
    return _instance;
  }
};

} // namespace geantx

#include "Geant/core/MemoryPool.tcpp"
