//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "traits.hpp"

//===----------------------------------------------------------------------===//
//  test the MemoryPool directly
//
template <typename Type>
void test(const std::string &id)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  try {
    std::cout << "\nTesting \"" << id << "\" of size " << sizeof(Type) << "...\n"
              << std::endl;
    geantx::MemoryPool<Type> _pool;
    Type *p                   = _pool.alloc();
    geantx::DevicePtr<Type> o = _pool.get(p);
    assign(p, Type());
    _pool.transfer_to(geantx::device::gpu(), stream);
    _pool.transfer(cudaMemcpyDeviceToHost, stream);
    _pool.transfer_to(geantx::device::cpu(), stream);
    _pool.transfer(cudaMemcpyHostToDevice, stream);
    printf("  > host pointer = %p\n", static_cast<void *>(p));
    printf("  > device pointer = %p\n", static_cast<void *>(o));
    _pool.free(p);
    o = _pool.get(p);
    printf("  > freed host pointer = %p\n", static_cast<void *>(p));
    printf("  > freed device pointer = %p\n", static_cast<void *>(o));
  } catch (const std::exception &e) {
    std::cerr << "  MemoryPool for type \"" << id << "\" threw an exception:\n    "
              << e.what() << "\n"
              << std::endl;
  }
  cudaStreamDestroy(stream);
}

//===----------------------------------------------------------------------===//
//  test the overloaded new/delete operators
//
template <typename Type>
void test_operators(const std::string &id)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  try {
    std::cout << "\nTesting \"" << id << "\" of size " << sizeof(Type) << "...\n"
              << std::endl;
    Type *obj                   = new Type();
    geantx::DevicePtr<Type> dev = obj->device_ptr();
    obj->transfer_to(geantx::device::gpu(), stream);
    obj->transfer_to(geantx::device::cpu(), stream);
    printf("  > host pointer = %p\n", static_cast<void *>(obj));
    printf("  > device pointer = %p\n", static_cast<void *>(dev));
    delete obj;
  } catch (const std::exception &e) {
    std::cerr << "  MemoryPool for type \"" << id << "\" threw an exception:\n    "
              << e.what() << "\n"
              << std::endl;
  }
  cudaStreamDestroy(stream);
}

//===----------------------------------------------------------------------===//
//  execute testing
//
int main()
{
  // catches SIGSEGV if occurs
  tim::enable_signal_detection();

  test<geantx::TrackState>("TrackState");
  test<geantx::OffloadTrackStatePinned>("Offloaded TrackState (Pinned)");
  test<geantx::OffloadTrackStateHost>("Offloaded TrackState (Host)");

  test_operators<geantx::OffloadTrackStatePinned>("Offloaded TrackState (Pinned)");
  test_operators<geantx::OffloadTrackStateHost>("Offloaded TrackState (Host)");

  std::cout << std::endl;
}
