//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file examples/core/memory/traits.hpp
 * @brief Trait definitions
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/MemoryPool.hpp"
#include "Geant/particles/Electron.hpp"
#include "Geant/particles/Gamma.hpp"
#include "Geant/particles/Neutron.hpp"
#include "Geant/track/TrackState.hpp"
#include "timemory/signal_detection.hpp"

namespace geantx {
// declarations
struct OffloadTrackStateHost;
struct OffloadTrackStatePinned;

// customization
template <>
struct OffloadMemoryPool<OffloadTrackStatePinned> : std::true_type {
};

template <>
struct OffloadMemoryPool<OffloadTrackStateHost> : std::true_type {
};

template <>
struct OffloadMemoryType<OffloadTrackStateHost> {
  using type = memory::host;
};

// create the types
struct OffloadTrackStateHost : public TrackState,
                               public MemoryPoolAllocator<OffloadTrackStateHost> {
};

struct OffloadTrackStatePinned : public TrackState,
                                 public MemoryPoolAllocator<OffloadTrackStatePinned> {
};

} // namespace geantx

template <typename Type,
          std::enable_if_t<std::is_move_constructible<Type>::value, int> = 0>
void assign(Type *ptr, Type &&obj)
{
  printf("  > assigning via move...\n");
  *ptr = std::move(obj);
}

template <typename Type,
          std::enable_if_t<!std::is_move_constructible<Type>::value, int> = 0>
void assign(Type *ptr, Type &&obj)
{
  printf("  > assigning via forward...\n");
  *ptr = std::forward<Type>(obj);
}
