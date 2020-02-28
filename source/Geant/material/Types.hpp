
#pragma once

/**
 * @brief   Generic vector and map containers.
 * @class   Types
 * @author  M Novak
 * @date    March 2017
 *
 * Makes possible to switch between vecgeom and std containers with a
 * -DUSE_VECGEOM_CONTAINERS cmake option.
 *
 */
#include "Geant/core/Config.hpp"

#ifdef USE_VECGEOM_CONTAINERS
#  include "VecGeom/base/Map.h"
#  include "VecGeom/base/Vector.h"
#else
#  include <map>
#  include <vector>
#endif

namespace geantx {
#ifdef USE_VECGEOM_CONTAINERS
template <typename T>
using Vector_t = vecgeom::Vector<T>;
template <typename KeyT, typename ValueT>
using Map_t = vecgeom::map<KeyT, ValueT>;
#else
template <typename T>
using Vector_t = std::vector<T>;
template <typename KeyT, typename ValueT>
using Map_t = std::map<KeyT, ValueT>;
#endif

} // namespace geantx
