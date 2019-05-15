
#include "Geant/core/Common.hpp"
#include "Geant/core/MemoryPool.hpp"
#include "Geant/particles/Electron.hpp"
#include "Geant/particles/Gamma.hpp"
#include "Geant/particles/Neutron.hpp"
#include "Geant/track/TrackState.hpp"

namespace geantx
{
using namespace geantphysics;

template <>
struct OffloadMemoryPool<Electron> : std::true_type
{
};
}

template <typename Type>
void test(const std::string& id)
{
    try
    {
        std::cout << "\nTesting \"" << id << "\" of size " << sizeof(Type) << "...\n" << std::endl;
        geantx::MemoryPool<Type> _pool;
        Type*                    p = _pool.alloc();
        Type*                    o = _pool.get(p);
        printf("  > host pointer = %p\n", static_cast<void*>(p));
        printf("  > device pointer = %p\n", static_cast<void*>(o));
        _pool.free(p);
        o = _pool.get(p);
        printf("  > freed host pointer = %p\n", static_cast<void*>(p));
        printf("  > freed device pointer = %p\n", static_cast<void*>(o));
    }
    catch(const std::exception& e)
    {
        std::cerr << "  MemoryPool for particle type \"" << id << "\" threw an exception:\n    " << e.what() << "\n"
                  << std::endl;
    }
}

int main()
{
    test<geantx::Electron>(geantx::Electron::Definition()->GetName());
    test<geantx::Gamma>(geantx::Gamma::Definition()->GetName());
    test<geantx::Neutron>(geantx::Neutron::Definition()->GetName());
    test<geantx::TrackState>("TrackState");

    std::cout << std::endl;
}
