//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "Geant/core/MemoryPool.hpp"
#include "Geant/particles/Electron.hpp"
#include "Geant/particles/Gamma.hpp"
#include "Geant/particles/Neutron.hpp"
#include "Geant/track/TrackState.hpp"
#include "timemory/signal_detection.hpp"

namespace geantx
{
// declarations
struct OffloadTrackStateHost;
struct OffloadTrackStatePinned;

// customization
template <>
struct OffloadMemoryPool<OffloadTrackStatePinned> : std::true_type
{
};

template <>
struct OffloadMemoryPool<OffloadTrackStateHost> : std::true_type
{
};

template <>
struct OffloadMemoryType<OffloadTrackStateHost>
{
    using type = memory::host;
};

// create the types
struct OffloadTrackStateHost
: public TrackState
, public MemoryPoolAllocator<OffloadTrackStateHost>
{
};

struct OffloadTrackStatePinned
: public TrackState
, public MemoryPoolAllocator<OffloadTrackStatePinned>
{
};

}  // namespace geantx

template <typename Type,
          std::enable_if_t<std::is_move_constructible<Type>::value, int> = 0>
void assign(Type* ptr, Type&& obj)
{
    printf("  > assigning via move...\n");
    *ptr = std::move(obj);
}

template <typename Type,
          std::enable_if_t<!std::is_move_constructible<Type>::value, int> = 0>
void assign(Type* ptr, Type&& obj)
{
    printf("  > assigning via forward...\n");
    *ptr = std::forward<Type>(obj);
}

template <typename Type>
void test(const std::string& id)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    try
    {
        std::cout << "\nTesting \"" << id << "\" of size " << sizeof(Type) << "...\n"
                  << std::endl;
        geantx::MemoryPool<Type> _pool;
        Type*                    p = _pool.alloc();
        Type*                    o = _pool.get(p);
        assign(p, Type());
        _pool.transfer_to(geantx::device::gpu(), stream);
        _pool.transfer(cudaMemcpyDeviceToHost, stream);
        _pool.transfer_to(geantx::device::cpu(), stream);
        _pool.transfer(cudaMemcpyHostToDevice, stream);
        printf("  > host pointer = %p\n", static_cast<void*>(p));
        printf("  > device pointer = %p\n", static_cast<void*>(o));
        _pool.free(p);
        o = _pool.get(p);
        printf("  > freed host pointer = %p\n", static_cast<void*>(p));
        printf("  > freed device pointer = %p\n", static_cast<void*>(o));
    }
    catch(const std::exception& e)
    {
        std::cerr << "  MemoryPool for type \"" << id << "\" threw an exception:\n    "
                  << e.what() << "\n"
                  << std::endl;
    }
    cudaStreamDestroy(stream);
}

template <typename Type>
void test_operators(const std::string& id)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    try
    {
        std::cout << "\nTesting \"" << id << "\" of size " << sizeof(Type) << "...\n"
                  << std::endl;
        Type* obj = new Type();
        Type* dev = obj->device_ptr();
        obj->transfer_to(geantx::device::gpu(), stream);
        obj->transfer_to(geantx::device::cpu(), stream);
        printf("  > host pointer = %p\n", static_cast<void*>(obj));
        printf("  > device pointer = %p\n", static_cast<void*>(dev));
        delete obj;
    }
    catch(const std::exception& e)
    {
        std::cerr << "  MemoryPool for type \"" << id << "\" threw an exception:\n    "
                  << e.what() << "\n"
                  << std::endl;
    }
    cudaStreamDestroy(stream);
}

int main()
{
    tim::enable_signal_detection();

    // test<geantx::Electron>(geantx::Electron::Definition()->GetName());
    // test<geantx::Gamma>(geantx::Gamma::Definition()->GetName());
    // test<geantx::Neutron>(geantx::Neutron::Definition()->GetName());
    test<geantx::TrackState>("TrackState");
    test<geantx::OffloadTrackStatePinned>("Offloaded TrackState (Pinned)");
    test<geantx::OffloadTrackStateHost>("Offloaded TrackState (Host)");

    test_operators<geantx::OffloadTrackStatePinned>("Offloaded TrackState (Pinned)");
    test_operators<geantx::OffloadTrackStateHost>("Offloaded TrackState (Host)");

    std::cout << std::endl;
}
