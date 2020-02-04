//
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
//
/**
 * @file
 * @brief
 */
//===----------------------------------------------------------------------===//
//

#pragma once

#include "Geant/core/Logger.hpp"
#include "Geant/core/Memory.hpp"
#include "Geant/core/MemoryPool.hpp"
#include "Geant/core/Tuple.hpp"
#include "Geant/track/TrackState.hpp"

#include <deque>
#include <list>
#include <map>
#include <tuple>
#include <vector>

#include <timemory/timemory.hpp>

using namespace geantx;

//===----------------------------------------------------------------------===//
struct Track;

namespace geantx {
template <>
struct OffloadMemoryPool<Track> : std::true_type {};
template <>
struct OffloadMemoryType<Track> {
  using type = memory::host;
};
} // namespace geantx

// create the types
struct Track : public TrackState, public MemoryPoolAllocator<Track, true> {};

//===----------------------------------------------------------------------===//

struct TrackManager {
  using this_type           = TrackManager;
  using TrackManagerArray_t = std::vector<Track *>;
  using TrackSpecialized_t  = std::function<bool(Track *)>;
  using size_type           = typename TrackManagerArray_t::size_type;

  size_type m_stride = 1;
  TrackManagerArray_t m_tracks;
  TrackManager *m_generic             = nullptr;
  TrackSpecialized_t m_is_specialized = [](Track *) { return false; };

  TrackManager(size_type n = 1) : m_stride(n) { m_tracks.resize(n, nullptr); }

  ~TrackManager()
  {
    for (auto &itr : m_tracks)
      delete itr;
  }

  bool IsGeneric() const { return m_generic == nullptr; }

  void SetGeneric(TrackManager *_manager)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (_manager != this) m_generic = _manager;
  }

  template <typename _Func>
  void SetSpecialized(_Func &&_func)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (m_generic) m_is_specialized = std::forward<_Func>(_func);
  }

  Track *PushTrack(Track *_track, size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (!_track) return nullptr;

    if (!IsAlive(*_track)) {
      delete _track;
      return nullptr;
    }

    for (size_type i = offset; i < m_tracks.size(); i += m_stride) {
      if (!m_tracks[i]) {
        m_tracks[i] = _track;
        return nullptr;
      }
    }
    auto idx = m_tracks.size();
    Resize(idx + m_stride);
    m_tracks[idx] = _track;
    return nullptr;
  }

  Track *PopTrack(size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    Track *_track = nullptr;
    for (size_type i = offset; i < m_tracks.size(); i += m_stride) {
      if (m_tracks[i]) {
        _track      = m_tracks[i];
        m_tracks[i] = nullptr;
        return _track;
      }
    }
    return _track;
  }

  template <typename T>
  Track *PushTrack(Track *_track, size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    // generic will have
    if (m_generic == nullptr) {
      return PushTrack(_track, offset);
    } else {
      return (m_is_specialized(_track)) ? PushTrack(_track, offset)
                                        : m_generic->PushTrack(_track, offset);
    }
  }

  template <typename T>
  Track *PopTrack(size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return Empty() ? nullptr : PopTrack(offset);
  }

  bool Empty() const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    for (size_type i = 0; i < m_tracks.size(); ++i)
      if (m_tracks[i]) return false;
    return true;
  }

  size_type Size() const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    size_type sz = 0;
    for (size_type i = 0; i < m_tracks.size(); ++i) {
      if (m_tracks[i]) ++sz;
    }
    return sz;
  }

  void Resize(size_type n)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (m_tracks.size() < n) m_tracks.resize(n);
  }

  template <size_t _N>
  static TrackManager *Instance()
  {
    static thread_local auto _instance = std::make_unique<TrackManager>();
    return _instance.get();
  }
};

//===----------------------------------------------------------------------===//

template <typename... ParticleTypes>
struct VariadicTrackManager {
  using type_list = std::tuple<ParticleTypes...>;
  using this_type = VariadicTrackManager<ParticleTypes...>;
  using TupleType = std::tuple<ParticleTypes...>;

  // the size of type list
  static constexpr std::size_t num_types = sizeof...(ParticleTypes);
  static constexpr std::size_t size      = num_types + 1;

  // array of particle-specific track-managers + a generic at the end
  using TrackManagerArray_t = std::array<TrackManager *, size>;
  using size_type           = typename TrackManagerArray_t::size_type;

  template <typename _Tp>
  struct GetIndexOf {
    static constexpr auto value = index_of<_Tp, TupleType>::value;
  };

  //----------------------------------------------------------------------------------//
  //  meta-programming recursion helpers
  //
  struct _Impl {
    template <size_t _I, size_t... _Idx, enable_if_t<(sizeof...(_Idx) == 0), char> = 0>
    static bool Empty(const TrackManagerArray_t &_managers,
                      std::index_sequence<_I, _Idx...>)
    {
      // if manager at given index is empty
      return std::get<_I>(_managers)->Empty();
    }

    template <size_t _I, size_t... _Idx,
              std::enable_if_t<(sizeof...(_Idx) > 0), char> = 0>
    static bool Empty(const TrackManagerArray_t &_managers,
                      std::index_sequence<_I, _Idx...>)
    {
      // recursive determination if manager is empty
      return std::get<_I>(_managers)->Empty() &&
             Empty<_Idx...>(_managers, std::index_sequence<_Idx...>{});
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), char> = 0>
    static void Specialize(TrackManagerArray_t &_managers)
    {
      // generate the functor that determines whether a track meets the
      // specialized criterion
      constexpr auto _I = index_of<_Tp, type_list>::value;
      _managers[_I]->SetSpecialized([](Track *_t) {
        return (!_t) ? false : (_t->fPhysicsState.fParticleDefId == _Tp::PdgCode);
      });
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), char> = 0>
    static void Specialize(TrackManagerArray_t &_managers)
    {
      // recursive specialization
      Specialize<_Tp>(_managers);
      Specialize<_Tail...>(_managers);
    }

    template <size_t _I, size_t... _Idx, enable_if_t<(sizeof...(_Idx) == 0), char> = 0>
    static size_type Size(const TrackManagerArray_t &_managers,
                          std::index_sequence<_I, _Idx...>)
    {
      // size of manager at given index
      return std::get<_I>(_managers)->Size();
    }

    template <size_t _I, size_t... _Idx,
              std::enable_if_t<(sizeof...(_Idx) > 0), char> = 0>
    static size_type Size(const TrackManagerArray_t &_managers,
                          std::index_sequence<_I, _Idx...>)
    {
      // recursive addition of all the sizes
      return std::get<_I>(_managers)->Size() +
             Size<_Idx...>(_managers, std::index_sequence<_Idx...>{});
    }
  };

  //----------------------------------------------------------------------------------//
  //  generates the specialized lambdas
  //
  struct _SpecializedImpl {};

  TrackManagerArray_t m_tracks;

  VariadicTrackManager(TrackManager *_generic)
  {
    m_tracks[num_types] = _generic;

    // the pointer to last instance (generic) could actually be assigned
    // to standard track manager
    for (size_t i = 0; i < num_types; ++i) {
      m_tracks[i] = new TrackManager();
      m_tracks[i]->SetGeneric(m_tracks[num_types]);
    }
    _Impl::template Specialize<ParticleTypes...>(m_tracks);
  }

  ~VariadicTrackManager()
  {
    for (size_t i = 0; i < num_types; ++i)
      delete m_tracks[i];
  }

  // without template params, we push to generic at end
  Track *PushTrack(Track *_track, size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    // geantx::Log(kInfo) << GEANT_HERE << "pushing track: " << *_track;
    return m_tracks[num_types]->PushTrack(_track, offset);
  }

  // with template params, we push to particle-specific queue
  template <typename ParticleType,
            std::enable_if_t<(is_one_of<ParticleType, TupleType>::value), int> = 0>
  Track *PushTrack(Track *_track, size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    // stopped particles should not be specialized since they require
    // no transport steps
    // if(IsStopped(*_track))
    //    return m_tracks[num_types]->PushTrack(_track, offset);
    return m_tracks[GetIndexOf<ParticleType>::value]->PushTrack(_track, offset);
  }

  // if the particle type is provided in template parameter and but not specialized
  // then re-direct to generic (non-template) PushTrack
  template <typename ParticleType,
            std::enable_if_t<!(is_one_of<ParticleType, TupleType>::value), int> = 0>
  Track *PushTrack(Track *_track, size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    if (_track->fPhysicsState.fParticleDefId == 0)
      _track->fPhysicsState.fParticleDefId = ParticleType::PdgCode;
    return PushTrack(_track, offset);
  }

  // without template params, we pop from generic manager at end
  Track *PopTrack(size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    TrackManager *&_manager = m_tracks[num_types];
    return _manager->PopTrack(offset);
  }

  // if particle type is provided in template and one of the specializations,
  // pop from the specialized manager
  template <typename ParticleType,
            std::enable_if_t<(is_one_of<ParticleType, TupleType>::value), int> = 0>
  Track *PopTrack(size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    TrackManager *&_manager = m_tracks[GetIndexOf<ParticleType>::value];
    return _manager->PopTrack(offset);
  }

  // if particle type is provided in template but not one of the specializations,
  // re-direct to non-template function that uses the generic manager
  template <typename ParticleType,
            std::enable_if_t<!(is_one_of<ParticleType, TupleType>::value), int> = 0>
  Track *PopTrack(size_type offset = 0)
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return PopTrack(offset);
  }

  // check if all managers are empty
  bool Empty() const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return _Impl::Empty(m_tracks, std::make_index_sequence<num_types>{});
  }

  // get sum of all tracks in all (specialized + generic) managers
  size_type Size() const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return _Impl::Size(m_tracks, std::make_index_sequence<num_types>{});
  }

  // get the size of particular specialization
  template <typename ParticleType,
            std::enable_if_t<(is_one_of<ParticleType, TupleType>::value), int> = 0>
  size_type Size() const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return m_tracks[GetIndexOf<ParticleType>::value]->Size();
  }

  // get the size of generic when the specialization is not valid
  template <typename ParticleType,
            std::enable_if_t<!(is_one_of<ParticleType, TupleType>::value), int> = 0>
  size_type Size() const
  {
    GEANT_THIS_TYPE_TESTING_MARKER("");
    return 0;
  }
};
