
#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/VectorTypes.hpp"
#include "Geant/core/Macros.hpp"

#ifdef VECCORE_CUDA
#include "base/Vector.h"
template <class T>
using vector_t = vecgeom::Vector<T>;
#else
#include <vector>
#ifdef GEANT_USE_NUMA
#include <GeantNuma.h>
template <class T>
using vector_t = std::vector<T, geantx::NumaAllocator<T>>;
#else
template <class T>
using vector_t = std::vector<T>;
#endif
#endif

// three vector types
using ThreeVector = vecgeom::Vector3D<double>;
template <typename T>
using Vector3D = vecgeom::Vector3D<T>;

namespace geantphysics {
class Particle;

enum EModel_t {
  kMSC = GEANT_BIT(0)
  // others to be added
};
} // namespace geantphysics
typedef geantphysics::Particle Particle_t;

#include "navigation/NavigationState.h"
typedef VECGEOM_NAMESPACE::NavigationState VolumePath_t;
#include "Geant/material/Material.hpp"
typedef geantphysics::Material Material_t;
#include "volumes/LogicalVolume.h"
typedef VECGEOM_NAMESPACE::LogicalVolume Volume_t;
#include "volumes/PlacedVolume.h"
typedef VECGEOM_NAMESPACE::VPlacedVolume Node_t;

namespace geantx {
inline namespace cuda {
template <typename _Tp>
using device_info = std::unordered_map<int, _Tp>;

}
} // namespace geant
