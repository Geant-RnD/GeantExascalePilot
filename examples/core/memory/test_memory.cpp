
#include "Geant/core/Common.hpp"
#include "Geant/core/MemoryPool.hpp"
#include "Geant/particles/Electron.hpp"
#include "Geant/particles/Gamma.hpp"
#include "Geant/particles/Neutron.hpp"
#include "Geant/track/TrackState.hpp"
#include "timemory/signal_detection.hpp"

namespace geantx
{
using namespace geantphysics;

struct OffloadTrackState : public TrackState
{
};

template <>
struct OffloadMemoryPool<Electron> : std::true_type
{
};

template <>
struct OffloadMemoryPool<OffloadTrackState> : std::true_type
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
        Type                     _p();
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
        std::cerr << "  MemoryPool for particle type \"" << id
                  << "\" threw an exception:\n    " << e.what() << "\n"
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
    test<geantx::OffloadTrackState>("Offloaded TrackState");

    std::cout << std::endl;
}
