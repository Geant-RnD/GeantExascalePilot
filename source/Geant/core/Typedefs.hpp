
#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/Macros.hpp"
#include "Geant/core/VectorTypes.hpp"
#include "Geant/material/Material.hpp"
#include "navigation/NavigationState.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"

#ifdef VECCORE_CUDA
#include "base/Vector.h"
#else
#include <vector>
#ifdef GEANT_USE_NUMA
#include <GeantNuma.h>
#endif
#endif

#include <unordered_map>
#include <vector>

// XXX: move to geantx namespace
namespace geantphysics {
class Particle;

enum EModel_t {
  kMSC = GEANT_BIT(0)
  // others to be added
};
} // namespace geantphysics

namespace geantx {
typedef VECGEOM_NAMESPACE::NavigationState VolumePath_t;
typedef geantphysics::Material Material_t;
typedef VECGEOM_NAMESPACE::LogicalVolume Volume_t;
typedef VECGEOM_NAMESPACE::VPlacedVolume Node_t;

#ifdef VECCORE_CUDA
template <class T>
using vector_t = vecgeom::Vector<T>;
#else
#ifdef GEANT_USE_NUMA
template <class T>
using vector_t = std::vector<T, geantx::NumaAllocator<T>>;
#else
template <class T>
using vector_t = std::vector<T>;
#endif
#endif

// three vector types
template <typename T>
using Vector3D    = vecgeom::Vector3D<T>;
using ThreeVector = Vector3D<double>;

typedef geantphysics::Particle Particle_t;

inline namespace cuda {
template <typename _Tp>
using device_info = std::unordered_map<int, _Tp>;
}
} // namespace geantx

// XXX: remove me once namespaces are fixed
using geantx::Particle_t;
using geantx::ThreeVector;
using geantx::Vector3D;
