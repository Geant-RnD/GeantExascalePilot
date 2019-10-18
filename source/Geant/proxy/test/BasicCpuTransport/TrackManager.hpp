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

using Track = TrackState;

template <typename ParticleType>
struct TrackCaster : public TrackState
{};

//===----------------------------------------------------------------------===//

struct TrackManager
{
    using TrackManagerArray = std::list<Track*>;
    TrackManagerArray m_tracks;

    TrackManager() = default;
    ~TrackManager()
    {
        for(auto& itr : m_tracks) delete itr;
    }

    void PushTrack(Track* _track)
    {
        geantx::Log(kInfo) << GEANT_HERE << "pushing track: " << *_track;
        m_tracks.emplace_front(_track);
    }
    Track* PopTrack()
    {
        Track* _track = nullptr;
        auto   sz     = m_tracks.size();
        if(sz > 0)
        {
            _track = m_tracks.back();
            m_tracks.resize(sz - 1);
        }
        if(_track) geantx::Log(kInfo) << GEANT_HERE << "popping track: " << *_track;
        return _track;
    }

    template <template <typename, typename...> class _Container, typename... _Extra>
    void PopTrack(_Container<Track*, _Extra...>& _arr)
    {
        Track* _track = nullptr;
        auto   sz     = m_tracks.size();
        auto   nget   = (_arr.size() == 0) ? sz : _arr.size();
        auto   nrem   = sz - nget;
        auto   _beg   = m_tracks.rbegin();
        auto   _end   = m_tracks.rend();
        std::advance(_beg, nrem);
        _arr.assign(_beg, _end);
        m_tracks.resize(nrem);
    }
};

//===----------------------------------------------------------------------===//

template <typename... ParticleTypes>
struct VariadicTrackManager
{
    // the size of type list
    static constexpr std::size_t num_types = sizeof...(ParticleTypes);
    using TupleType                        = std::tuple<ParticleTypes...>;

    // array of particle-specific track-managers + a generic at the end
    using TrackManagerArray = std::array<TrackManager*, num_types + 1>;

    TrackManagerArray m_tracks;

    VariadicTrackManager()
    {
        // the pointer to last instance (generic) could actually be assigned
        // to standard track manager
        for(auto& itr : m_tracks) itr = new TrackManager();
    }

    ~VariadicTrackManager()
    {
        for(auto& itr : m_tracks) delete itr;
    }

    // without template params, we push to generic at end
    void PushTrack(Track* _track)
    {
        geantx::Log(kInfo) << GEANT_HERE << "pushing track: " << *_track;
        m_tracks[num_types]->PushTrack(_track);
    }

    // with template params, we push to particle-specific queue
    template <typename ParticleType,
              std::enable_if_t<(is_one_of<ParticleType, TupleType>::value), int> = 0>
    void PushTrack(Track* _track)
    {
        geantx::Log(kInfo) << GEANT_HERE << "pushing track: " << *_track;
        m_tracks[index_of<ParticleType, TupleType>::value]->PushTrack(_track);
    }

    template <typename ParticleType,
              std::enable_if_t<!(is_one_of<ParticleType, TupleType>::value), int> = 0>
    void PushTrack(Track* _track)
    {
        geantx::Log(kInfo) << GEANT_HERE << "pushing track: " << *_track;
        PushTrack(_track);
    }

    // without template params, we pop from end
    Track* PopTrack()
    {
        TrackManager*& _manager = m_tracks[num_types];
        Track*         _track   = _manager->PopTrack();
        if(_track) geantx::Log(kInfo) << GEANT_HERE << "popping track: " << *_track;
        return _track;
    }

    template <typename ParticleType,
              std::enable_if_t<(is_one_of<ParticleType, TupleType>::value), int> = 0>
    TrackCaster<ParticleType>* PopTrack()
    {
        using Caster_t          = TrackCaster<ParticleType>;
        TrackManager*& _manager = m_tracks[index_of<ParticleType, TupleType>::value];
        Track*         _track   = _manager->PopTrack();
        if(_track) geantx::Log(kInfo) << GEANT_HERE << "popping track: " << *_track;
        return static_cast<Caster_t*>(_track);
    }

    template <typename ParticleType,
              std::enable_if_t<!(is_one_of<ParticleType, TupleType>::value), int> = 0>
    Track* PopTrack()
    {
        return PopTrack();
    }
};
