# Transport Design Proposal

## Overview

This document lays out a possible method of designing the transportation engine
for the GeantExascaleProxy. In general, this design scheme proposes to use
a template library as the "glue". This specification does not imply that the
entire library be required to templated in the slightest.

## Dynamic vs Runtime Polymorphism

Consider the following definition below using dynamic polymorphism:

```c++
struct ParticleDefinition
{
    ParticleDefinition() {}
    virtual string GetName() const = 0;
};

struct Electron : public ParticleDefinition
{
    Electron() {}
    virtual string GetName() const { return "e-"; }
};
```

This could easily be converted to static polymorphism via:

```c++
template <typename Type>
struct ParticleDefinition
{
    ParticleDefinition() {}
    string GetName() const { return static_cast<Type*>(this)->GetName(); }
};

struct Electron : public ParticleDefinition<Electron>
{
    Electron() {}
    // can stay the same
    virtual string GetName() const { return "e-"; }
};
```

The problem is that everything using `ParticleDefinition*` must now be templated
to use `ParticleDefinition<Type>` which commonly leads to the entire library being
templated, e.g.

```c++
struct Track
{
    intmax_t m_track_id;
    ParticleDefinition* m_pdef;

    ParticleDefinition* GetParticleDefinition() const { return m_pdef; }
    string GetParticleName() const { return m_pdef->GetName(); }
    intmax_t GetTrackId() const { return m_track_id; }
};
```

turns into:

```c++
template <typename Type>
struct Track
{
    using PDef_t = ParticleDefinition<Type>;

    intmax_t m_track_id;
    PDef_t* m_pdef;

    PDef_t* GetParticleDefinition() const { return m_pdef; }
    string GetParticleName() const { return m_pdef->GetName(); }
    intmax_t GetTrackId() const { return m_track_id; }
};
```

and track with different `Type` parameters can no longer be stored in the same container as such:

```c++
struct TrackManager
{
    std::deque<Track*> m_tracks;

    void PushTrack(Track* track) { m_tracks.push_back(track); }
    Track* PopTrack()
    {
        Track* track = m_tracks.front();
        m_tracks.pop_front();
        return track;
    }
};
```

The standard accommadation tends to be the following:

```c++
template <typename Type>
struct TrackManager
{
    using Track_t = Track<Type>;
    std::deque<Track_t*> m_tracks;

    void PushTrack(Track_t* track) { m_tracks.push_back(track); }
    Track_t* PopTrack()
    {
        Track_t* track = m_tracks.front();
        m_tracks.pop_front();
        return track;
    }
};
```

# Optional elimination of the virtual function table

Here is a alternative that enables a scheme to eliminate the virtual table on top of a dynamic polymorphism implementation only when desired/needed:

```c++
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
    Type*  GetParticleDefinition() const { return static_cast<Type*>(m_pdef); }
    std::string GetParticleName() const { return static_cast<Type*>(m_pdef)->GetName(); }
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

    // array of track managers
    TrackManagerArray m_tracks;

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
```

where `index_of` is defined as:

```c++
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
```

On it's surface, this may not appear to provide much but consider the following:

- The base types were not modified beyond a friend statement
  - Any the friend statement isn't _necessary_
  - Thus only when we want to do something type-specific, such as transport
  electrons on GPU, is there a need to start embedding types
- Invoking `VariadicTrackManager::PushTrack<ParticleType>(track)` required specifying `ParticleType`
  - This is something a dedicated particle process manager would know

## Example

```c++
using Manager_t = VariadicTrackManager<Electron, Proton>;

template <typename _Tp>
void print(const _Tp* t, int expected_id, const std::string& access_method)
{
    auto w = 24;
    std::cout << "track " << expected_id << " (" << access_method << "). name : " << std::setw(15)
              << t->GetParticleName() << ",  id : " << t->m_track_id << ",  typeid : " << std::setw(w)
              << typeid(*t).name() << ", particle def typeid : " << std::setw(10)
              << typeid(*t->GetParticleDefinition()).name() << std::endl;
}

int main()
{
    Electron* electron = new Electron();
    Proton* proton = new Proton();
    Track* t1 = new Track();
    Track* t2 = new Track();
    Track* t3 = new Track();

    t1->m_pdef = electron;
    t1->m_track_id = 0;
    t2->m_pdef = proton;
    t2->m_track_id = 1;
    t3->m_pdef = proton;
    t3->m_track_id = 2;

    Manager_t* manager = new Manager_t;

    manager->PushTrack<Electron>(t1);
    manager->PushTrack(t2);
    manager->PushTrack<Proton>(t3);

    auto _t1 = manager->PopTrack<Electron>();
    auto _t2 = manager->PopTrack();
    auto _t3 = manager->PopTrack<Proton>();

    print(t1, 0, "pushed");
    print(t2, 1, "pushed");
    print(t3, 2, "pushed");

    print(_t1, 0, "popped");
    print(_t2, 1, "popped");
    print(_t3, 2, "popped");

    delete manager;
    delete t1;
    delete t2;
    delete t3;
    delete electron;
    delete proton;
}
```

```shell
track 0 (pushed). name :              e-,  id : 0,  typeid :                   5Track, particle def typeid :  8Electron
track 1 (pushed). name :          proton,  id : 1,  typeid :                   5Track, particle def typeid :    6Proton
track 2 (pushed). name :          proton,  id : 2,  typeid :                   5Track, particle def typeid :    6Proton

track 0 (popped). name :       e-_casted,  id : 0,  typeid : 11TrackCasterI8ElectronE, particle def typeid :  8Electron
track 1 (popped). name :          proton,  id : 1,  typeid :                   5Track, particle def typeid :    6Proton
track 2 (popped). name :   proton_casted,  id : 2,  typeid :   11TrackCasterI6ProtonE, particle def typeid :    6Proton
```
