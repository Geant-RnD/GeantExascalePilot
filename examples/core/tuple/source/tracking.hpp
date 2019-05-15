
#include <array>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>

// base declaration
template <typename _Tp, typename Type>
struct index_of;

// when it matches (e.g. _Tp matches Head)
template <typename _Tp, typename... Types>
struct index_of<_Tp, std::tuple<_Tp, Types...>>
{
    static constexpr std::size_t value = 0;
};

// recursive addition to get index of type in tuple until "Head" matches "_Tp"
template <typename _Tp, typename Head, typename... Tail>
struct index_of<_Tp, std::tuple<Head, Tail...>>
{
    static constexpr std::size_t value = 1 + index_of<_Tp, std::tuple<Tail...>>::value;
};

struct ParticleDefinition
{
    ParticleDefinition() {}
    virtual std::string GetName() const = 0;
};

struct Electron : public ParticleDefinition
{
    Electron() {}
    virtual std::string GetName() const { return "e-"; }
};

struct Proton : public ParticleDefinition
{
    Proton() {}
    virtual std::string GetName() const { return "proton"; }
};

struct Track
{
    intmax_t            m_track_id;
    ParticleDefinition* m_pdef;

    ParticleDefinition* GetParticleDefinition() const { return m_pdef; }
    std::string         GetParticleName() const { return m_pdef->GetName(); }
    intmax_t            GetTrackId() const { return m_track_id; }
};

struct TrackManager
{
    std::deque<Track*> m_tracks;

    void   PushTrack(Track* track) { m_tracks.push_back(track); }
    Track* PopTrack()
    {
        if(m_tracks.empty())
            return nullptr;
        Track* track = m_tracks.front();
        m_tracks.pop_front();
        return track;
    }
};

// overload only the polymorphic functions
// NOTE:
//  this should be added to Track:
//
//      template <typename Type> friend class TrackCaster;
//
// to ensure is can properly modify any runtime polymorphism usage
//
template <typename Type>
struct TrackCaster : public Track
{
    Type*       GetParticleDefinition() const { return static_cast<Type*>(m_pdef); }
    std::string GetParticleName() const
    {
        return static_cast<Type*>(m_pdef)->Type::GetName() + "_casted";
    }
    // no need to reimplement GetTrackId()
};

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
        for(auto& itr : m_tracks)
        {
            itr = new TrackManager();
        }
    }

    ~VariadicTrackManager()
    {
        for(auto& itr : m_tracks)
        {
            delete itr;
        }
    }

    // without template params, we push to generic at end
    void PushTrack(Track* track) { m_tracks[num_types]->PushTrack(track); }

    // with template params, we push to particle-specific queue
    template <typename ParticleType>
    void PushTrack(Track* track)
    {
        m_tracks[index_of<ParticleType, TupleType>::value]->PushTrack(track);
    }

    // without template params, we pop from end
    Track* PopTrack()
    {
        TrackManager*& _manager = m_tracks[num_types];
        Track*         track    = _manager->PopTrack();
        return track;
    }

    template <typename ParticleType>
    TrackCaster<ParticleType>* PopTrack()
    {
        using Caster_t          = TrackCaster<ParticleType>;
        TrackManager*& _manager = m_tracks[index_of<ParticleType, TupleType>::value];
        Track*         track    = _manager->PopTrack();
        return static_cast<Caster_t*>(track);
    }
};
