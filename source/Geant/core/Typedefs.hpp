
#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/VectorTypes.hpp"

#ifdef VECCORE_CUDA
#include "base/Vector.h"
template <class T>
using vector_t = vecgeom::Vector<T>;
#else
#include <vector>
#ifdef GEANT_USE_NUMA
#include <GeantNuma.h>
template <class T>
using vector_t = std::vector<T, geant::NumaAllocator<T>>;
#else
template <class T>
using vector_t = std::vector<T>;
#endif
#endif

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
