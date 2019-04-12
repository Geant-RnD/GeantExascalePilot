# Transport Design Proposal

## Overview

This document lays out a possible method of designing the transportation engine
for the GeantExascaleProxy. In general, this design scheme proposes to use
a template library as the "glue". This specification does not imply that the
entire library be required to templated but instead proposes a hybrid methodology.

### Dynamic vs Runtime Polymorphism

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

The standard accommodation tends to be the following:

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

### Optional Type Upcasting

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
    std::string GetParticleName() const
    {
        return static_cast<Type*>(m_pdef)->GetName() + "_casted";
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
        // the pointer to last instance (generic) could actually be assigned
        // to standard track manager
        for(auto& itr : m_tracks)
            itr = new TrackManager();
    }

    ~VariadicTrackManager() { }
    {
        for(auto& itr : m_tracks)
            delete itr;
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
```

where `index_of` is defined as:

```c++
// base declaration
template <typename T, typename Type>
struct index_of;

// when it matches (e.g. Type matches Head)
template <typename Type, typename... Types>
struct index_of<Type, std::tuple<Type, Types...>>
{
    static constexpr std::size_t value = 0;
};

// recursive addition to get index of type in tuple until "Head" matches "Type"
template <typename Type, typename Head, typename... Tail>
struct index_of<Type, std::tuple<Head, Tail...>>
{
    static constexpr std::size_t value = 1 + index_of<Type, std::tuple<Tail...>>::value;
};
```

On it's surface, this may not appear to provide much but consider the following:

- The base types were not modified beyond a friend statement
  - The friend statement is not _necessary_
  - Thus only when we want to do something type-specific, such as transport
  electrons on GPU, is there a need to start embedding types
- Invoking `VariadicTrackManager::PushTrack<ParticleType>(track)` required specifying `ParticleType`
  - This is something a dedicated particle process manager would know

### Example

```c++
using Manager_t = VariadicTrackManager<Electron, Proton>;

template <typename Type>
void print(const Type* t, int expected_id, const std::string& access_method)
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

## Template Interface

In the previous section, a variadic template class was introduced on top of an existing class
designed to use dynamic polymorphism. This section will attempt to detail how such a layer can
be expanded in the grand scheme of execution.

There are many, many components to Monte Carlo transport so I will attempt to explain the
proposal by a simpler proxy and explain how they relate to GeantX.

### TiMemory

I developed the [TiMemory](https://github.com/jrmadsen/TiMemory) library as an easy way to
record timing and memory values and deltas in specific regions. I recently did an almost
complete re-write of the source code as a test-bed for the concepts I am now proposing.

Essentially, every measurement type executes the same series of routines (just as
a particle-process step does). It may help to make the following associations:

| Step | TiMemory | Monte Carlo Transport |
| ---- | -------- | --------------------- |
| construct | compute hash identifier | prepare to compute a step |
| `insert_node()` | determine call graph location | determine applicable processes for step |
| `set_prefix()` | optionally assign prefix to node | check conditions for step |
| `start()` | start recording something | select a process for step |
| `stop()` | stop recording something | propogate the step |
| destruct | add delta to call graph instance | handle secondaries |

#### Relevant Links

- [component_operations.hpp](https://github.com/jrmadsen/TiMemory/blob/graph-storage-redesign/source/timemory/component_operations.hpp)
- [component_tuple.hpp](https://github.com/jrmadsen/TiMemory/blob/graph-storage-redesign/source/timemory/component_tuple.hpp)
- [components.hpp](https://github.com/jrmadsen/TiMemory/blob/graph-storage-redesign/source/timemory/components.hpp)
- [auto_tuple.hpp](https://github.com/jrmadsen/TiMemory/blob/graph-storage-redesign/source/timemory/auto_tuple.hpp)
- [apply.hpp](https://github.com/jrmadsen/TiMemory/blob/graph-storage-redesign/source/timemory/apply.hpp)

#### `component_tuple`

The measurement types are held within a variadic
template class, much like the `VariadicTrackManager` above, called a `component_tuple`.
The responsibility of the `component_tuple` is to apply the generic member functions
such as `start()` and `stop()` to the instances in it's heterogeneous type list.

```c++
template <typename... Types>
class component_tuple
{
    static const std::size_t num_elements = sizeof...(Types);

public:
    using size_type   = intmax_t;
    using this_type   = component_tuple<Types...>;
    using data_t      = std::tuple<Types...>;
    using string_hash = std::hash<string_t>;
    using bool_array  = std::array<bool, num_elements>;

    component_tuple(const string_t& key, const string_t& tag = "cxx")
    {
        compute_identifier(key, tag);
        push();
    }

    ~component_tuple() { pop(); }

    void push();
    void pop();
    void start();
    void stop();
};
```

#### `auto_tuple`

The `component_tuple` instance is held by another variadic template class called
`auto_tuple`. The responsibility of the `auto_tuple` is to call the relevant functions
within `component_tuple` to achieve the desired functionality and it uses static type traits
to filter out types that are not available. In the context of TiMemory, this would
be PAPI events when compiled without PAPI support. In the context of GeantX, this filtering
would be similar to a manager class that only operated on the particle and physics
processes that were designated as available for GPU off-loading.

```c++
// trait that signifies that an implementation (e.g. PAPI) is available
template <typename Type> struct impl_available : std::true_type { };

#if !defined(TIMEMORY_USE_PAPI)

// overload specific to "papi_event" class when preprocessor disables support
template <> struct impl_available<papi_event> : std::false_type { };

#endif

// definition of auto_tuple
template <typename... Types>
class auto_tuple
{
public:
    using this_type    = auto_tuple<Types...>;

    // returning component_tuple<...> with type list only marked with:
    //   impl_available == std::true_type
    using object_type = type_filter<component::impl_available, component_tuple<Types...>>;

    // holds the filter component_tuple
    object_type m_temp_object;

    // below is not a real member function but demonstrates concept
    // that whatever functionality is desired is forwarded to "m_temp_object"
    // instance
    void start()
    {
        m_temp_object.init();
        m_temp_object.start();
    }
};
```

#### General sequence of operations

- Creation of `auto_tuple`
  - Computes a unique hash id from a thread-local reference counter and the label it is given (e.g. function and line where it was created)
  - Creates the `component_tuple`
    - Instructs all of the measurement instances to try to find a sibling or child node with a matching hash id
    - If a matching sibling or child node is found, the current node is assign to this match
    - If no matchin sibling or child node is found, the measurment is marked as new and assigned as the current node
    - Invokes an operator that evaluates whether the node is new, and if true, assigns a label to the node
  - Instructs `component_tuple` to start all measurements
- Destruction of `auto_tuple`
  - Instructs `component_tuple` to stop all measurments
    - All components add their delta into the graph-storage instance
    - All components set the current node in the cumulative graph storage to the parent of the current node


#### Base Static Polymorphic Struct

In the context of TiMemory, the base functionality of measurments are handled in a static polymorphic class. In the context of GeantX, objects such as below would:

- Be friends of polymorphic objects whose functionality they are providing on GPU
- Possibly handle memory copies
- Aggregate or select relevant data
- Layout memory in a desired structure for GPU (e.g. SOA)

```c++
//--------------------------------------------------------------------------------------//

template <typename Type, typename value_type = intmax_t>
struct base
{
    using this_type    = base<Type, value_type>;
    using storage_type = graph_storage<Type>;

    bool                            is_running   = false;
    bool                            is_transient = false;
    value_type                      value        = value_type();
    value_type                      accum        = value_type();
    intmax_t                        hashid       = 0;
    intmax_t                        laps         = 0;
    typename storage_type::iterator itr;

    //----------------------------------------------------------------------------------//
    // set the graph node prefix
    //
    void set_prefix(const string_t& _prefix)
    {
        storage_type::instance()->set_prefix(_prefix);
    }

    //----------------------------------------------------------------------------------//
    // insert the node into the graph
    //
    void insert_node(bool& exists, const intmax_t& _hashid)
    {
        hashid    = _hashid;
        Type& obj = static_cast<Type&>(*this);
        itr       = storage_type::instance()->insert(hashid, obj, exists);
    }

    //----------------------------------------------------------------------------------//
    // pop the node off the graph
    //
    void pop_node()
    {
        Type& obj = itr->obj();
        Type& rhs = static_cast<Type&>(*this);
        obj += rhs;
        obj.laps += rhs.laps;
        storage_type::instance()->pop();
    }

    //----------------------------------------------------------------------------------//
    //
    void start()
    {
        ++laps;
        static_cast<Type&>(*this).start();
    }

    //----------------------------------------------------------------------------------//
    //
    void stop() { static_cast<Type&>(*this).stop(); }

    // ....
};
```

#### Providing Specific Functionality to Base

In the context of TiMemory, these structs provide the implementation of the measurement.
In the context of GeantX, these structs would provide the implementation of the various
transport steps.

```c++
// the system's real time (i.e. wall time) clock, expressed as the amount of time since
// the epoch.
struct real_clock : public base<real_clock>
{
    using ratio_t    = std::nano;
    using value_type = intmax_t;
    using base_type  = base<real_clock, value_type>;

    static value_type record()
    {
        return tim::get_clock_real_now<intmax_t, ratio_t>();
    }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
        set_stopped();
    }
};

// this struct extracts the high-water mark (or a change in the high-water mark) of
// the resident set size (RSS). Which is current amount of memory in RAM
//
// when used on a system with swap enabled, this value may fluctuate but should not
// on an HPC system.
struct peak_rss : public base<peak_rss>
{
    using value_type = intmax_t;
    using base_type  = base<peak_rss, value_type>;

    static value_type  record()
    {
        return get_peak_rss();
    }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        auto tmp   = record();
        auto delta = tmp - value;
        accum      = std::max(accum, delta);
        value      = std::move(tmp);
        set_stopped();
    }
};
```

#### Providing Operations on Objects

In the context of TiMemory, there are certain measurments that "addition" is should mean
calculate the max of two objects instead of adding two values together, such as determining
the memory high-water mark change in a function. In the context of GeantX, there are
certain operations that are specific to process types and particle types and a similiar
strategy could be deployed.


```c++
template <typename Type> struct record_max : std::false_type { };

// mark "peak_rss" as using max instead of addition
template <> struct record_max<peak_rss> : std::true_type { };

//--------------------------------------------------------------------------------------//

namespace component
{

template <typename Type>
struct start
{
    start(Type& obj)
    {
        obj.start();
    }
};

//--------------------------------------------------------------------------------------//

template <typename Type>
struct print
{
    print(std::size_t _N, std::size_t _Ntot, const Type& obj, std::ostream& os,
          bool endline)
    {
        std::stringstream ss;
        ss << obj;
        if(_N + 1 < _Ntot)
            ss << ", ";
        else if(_N + 1 == _Ntot && endline)
            ss << std::endl;
        os << ss.str();
    }

//--------------------------------------------------------------------------------------//

template <typename Type>
struct plus
{
    // when marked with "record_max"
    template <typename _Up = Type, enable_if_t<(record_max<_Up>::value == true), int> = 0>
    plus(Type& obj, const Type& rhs)
    {
        obj = std::max(obj, rhs);
    }
    // when not marked with "record_max"
    template <typename _Up = Type, enable_if_t<(record_max<_Up>::value == false), int> = 0>
    plus(Type& obj, const Type& rhs)
    {
        obj += rhs;
    }

    plus(Type& obj, const intmax_t& rhs) { obj += rhs; }
};

} // end namespace component

```

Instead of type traits, we can explicitly customize the entire operator for a specific type, e.g.:

```c++
template <>
struct plus<record_max>
{
    plus(record_max& obj, const record_max& rhs)
    {
        obj = std::max(obj, rhs);
    }

    plus(record_max& obj, const intmax_t& rhs) { obj += rhs; }
};
```

#### Invoking Operations on Objects

In this section, I detail how the operations are applied onto the variadic template typelist.
At the end of the section, there is a sample of `apply<void>::access` called within
`component_tuple`. These implementations are what enables the heterogeneous types to
be called. It looks complex but they are extremely generic.

Essentially, here is how operations are invoked on a heterogenous type list using `apply<void>::access`:

- `apply<void>::access` takes a typelist as a template argument, e.g. `apply<void>::access<TypeList>`
- A tuple (of the heterogeneous types) is passed as the first argument
- The length of the typelist must match the length of the tuple
- `apply<void>::access` takes an unlimited number of arguments (including zero)
  after the first argument
- The size of the typelist/tuple is deduced by `apply<void>::access`
  - Let's call that value `N`
  - It calls an internal implementation that creates `N` function calls
  - In the expansion, at expansion #0, we have "`AccessType0`" and "`TupleObject0`" and so on
- For `0 .. N-1`, we call `AccessType0(TupleObject0, args...)`
  - Where `args...` are the remaining arguments
  - Using the defined types earlier, one such invocation would essentially expand to:

```c++
// assuming:
//      measurements = std::tuple < real_clock, peak_rss >;

real_clock & rc = std::get <0> (measurements);
peak_rss   & pr = std::get <1> (measurements);

component::start < real_clock > (rc);
component::start < peak_rss >   (pr);

// this is component::start definition above
template <typename Type>
struct component::start
{
    start(Type& obj)
    {
        obj.start();
    }
};
```

Within the `component_tuple`, the invocation above looks like the following:

```c++
template <typename... Types>
struct component_tuple
{
    // ...
    void start()
    {
        // component::start is the operator struct
        // this is the "TypeList"
        using apply_types = std::tuple< component::start <Types>... >;

        // "apply_types" (our operator typelist) expands to
        //
        //    std::tuple <
        //                component::start < real_clock >,
        //                component::start < peak_rss   >
        //               >;
        apply<void>::access<apply_types>(m_data);
    }
    // ...
};
```

As promised above, here is the actual implementation of `apply<void>::access`:

```c++
namespace internal
{
// skip this implementation but the generic implementation handles return types
template <typename _Ret> struct apply_impl { };

// for above operators which use constructors and thus do not return values
template <>
struct internal::apply_impl<void>
{
    // expansion when _N == _Nt
    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void apply_access(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(__args)...);
    }

    // expansion when _N < _Nt
    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
    static void apply_access(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(__args)...);
        // recursive call
        apply_access<_N + 1, _Nt, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }
};

} // end of internal namespace

template <typename _Ret> struct apply { };

template <>
struct apply<void>
{
    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void access(_Tuple&& __t, _Args&&... __args)
    {
        internal::apply_impl<void>::template apply_access<0, _N - 1, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }
};

```

## Summary

In the end, here is what a transport kernel implementation would theoretically resemble
based on this proposal:

```c++
//--------------------------------------------------------------------------------------//
// user-defined physics process types per-particle via specialization
//
template <>
struct PhysicsProcessesAvailable<Photon>
{
    using type = std::tuple<TransportProcess, ComptonProcess, PhotoElectProcess>;
}

//--------------------------------------------------------------------------------------//

template <intmax_t N, typename ParticleType, typename... Processes>
__global__
void ComputePIL(TrackCaster<ParticleType> tracks[N],
                PhysicsList<ParticleType, Processes...> physics[N],
                Geometry* geom)
{
    using Track_t = TrackCaster<ParticleType>;
    using PhysL_t = PhysicsList<ParticleType>;

    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < N; i += istride)
    {
        Track_t* track = tracks[i];
        PhysicsList* phys = physics[i];

        // operator struct for compute physical interaction length
        using ApplyTypes = std::tuple<geant::ComputePIL<Processes>...>;

        // assume something along the lines of PhysicsList::get()
        // returns a tuple of actual physics process instances specialized
        // on the particle type
        apply<void>::access<ApplyTypes>(phys->get(), track, geom);
    }
}

//--------------------------------------------------------------------------------------//

template <intmax_t N, typename ParticleType,
          typename... Processes, typename SecondaryTypes...>
__global__
void ApplyProcess(TrackCaster<ParticleType> tracks[N],
                  PhysicsList<ParticleType, Processes...> physics[N],
                  Geometry* geom,
                  TrackList<SecondaryTypes...>[N] secondaries)
{
    using Track_t = TrackCaster<ParticleType>;
    using PhysL_t = PhysicsList<ParticleType>;

    int i0      = blockIdx.x * blockDim.x + threadIdx.x;
    int istride = blockDim.x * gridDim.x;

    for(int i = i0; i < N; i += istride)
    {
        Track_t* track = tracks[i];
        PhysicsList* phys = physics[i];

        // this operator finds minimum PIL, completes step, and then
        // the process takes "secondaries" and does something like:
        //      secondaries->PushTrack<Photon>(secondary_photon);
        using ApplyTypes = std::tuple<geant::ApplyProcess<Processes>...>;

        apply<void>::access<ApplyTypes>(phys->get(), track, secondaries);
    }
    // ... etc.
}

//--------------------------------------------------------------------------------------//
// block implementation that ideally launches with warp size of 32
//
template <intmax_t N, typename ParticleType, typename... ParticleTypes>
auto DoStep(VariadicTrackManager<ParticleTypes...>* track_manager,
            Geometry* geom, CudaStream_t stream)
{
    // process types is a tuple
    using ProcessTypes = PhysicsProcessesAvailable<ParticleType>::type;

    // Each process defines a tuple of secondary particle types it can generate
    // "SecondaryTypelist":
    //    expands each proccess, gets secondary tuple, adds type to typelist
    //    unless it already exists
    using SecondaryTypes = SecondaryTypelist<ProcessTypes>::type;

    // assume these are filled/allocated/mem-copied
    TrackCaster<ParticleType> tracks[N];
    PhysicsList<ParticleType, Processes...> physics[N];

    // track_manager has member function returning array of pointers
    // to secondary queue
    //   when pushing back a particle transported on GPU, it does not copy memory
    //   when pushing back a particle transported on CPU, it handles memcpy
    TrackList<SecondaryTypes...>[N] secondaries =
        track_manager->GetSecondaryQueues<SecondaryTypes...>();

    // assume copy max of N tracks

    // launch maybe N blocks and and 1 threads or maybe 1 block and 32 threads
    // or maybe do some calculation based on N. Separating into different blocks
    // with 1 thread theoretically has the benefit that we don't have to be concerned
    // about thread divergence in a warp but would be very inefficient within a block
    // ... will be evaluated and considered later
    ComputePIL<N><<<1, N, 0, stream>>>(tracks, physics, geom);
    ApplyProcess<N><<1, N, 0, stream>>(tracks, physics, geom, secondaries);
}

//--------------------------------------------------------------------------------------//
// general implementation that launches kernels until all ready and secondary are
// finished
//
template <typename ParticleType, typename... ParticleTypes>
auto DoStep(VariadicTrackManager<ParticleTypes...>* track_manager,
            Geometry* geom, CudaStream_t stream)
{
    constexpr intmax_t max_launch = 32;
    while(track_manager->GetReadyQueue<ParticleType>().size() > 0)
    {
        intmax_t nready = track_manager->GetReadyQueue<ParticleType>().size();
        intmax_t n = std::min(max_launch, nready);

        // could probably achieve this with template unrolling and function forwarding
        switch (n)
        {
            case max_launch:
                DoStep<max_launch>(track_manager, geom, ngrid, nblock, stream);
                break;
            // ...
            // ...
            // ...
            case 1:
            default:
                DoStep<1>(track_manager, geom, ngrid, nblock, stream);
                break;
        }
        cudaStreamSynchronize(stream);
        track_manager->CopySecondaryToReady<ParticleType>();
    }
}

```