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
 * @brief Declaration of physics related typedefs.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/VectorTypes.hpp"
#include "Geant/material/Material.hpp"
#include "VecGeom/navigation/NavigationState.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"

#ifdef GEANT_CUDA
#  include "VecGeom/base/Vector.h"
#else
#  include <vector>
#  ifdef GEANT_USE_NUMA
#    include <GeantNuma.h>
#  endif
#endif

#include <unordered_map>
#include <vector>

// XXX: move to geantx namespace
namespace geantx {
class Particle;

enum EModel_t {
  kMSC = GEANT_BIT(0)
  // others to be added
};
} // namespace geantx

namespace geantx {
typedef VECGEOM_NAMESPACE::NavigationState VolumePath_t;
typedef VECGEOM_NAMESPACE::LogicalVolume Volume_t;
typedef VECGEOM_NAMESPACE::VPlacedVolume Node_t;

#ifdef GEANT_CUDA
template <typename T>
using vector_t = vecgeom::Vector<T>;
#else
#  ifdef GEANT_USE_NUMA
template <typename T>
using vector_t = std::vector<T, geantx::NumaAllocator<T>>;
#  else
template <typename T>
using vector_t = std::vector<T>;
#  endif
#endif

// three vector types
template <typename T>
using Vector3D    = vecgeom::Vector3D<T>;
using ThreeVector = Vector3D<double>;

typedef geantx::Particle Particle_t;

} // namespace geantx

// XXX: remove me once namespaces are fixed
using geantx::Particle_t;
using geantx::ThreeVector;
using geantx::Vector3D;
