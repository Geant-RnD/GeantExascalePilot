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

#include "Geant/core/Config.hpp"
#include "Geant/core/Logger.hpp"
#include "Geant/core/Memory.hpp"
#include "Geant/core/MemoryPool.hpp"
#include "Geant/proxy/ProxySystemOfUnits.hpp"
#include "Geant/geometry/UserDetectorConstruction.hpp"
#include "Geant/track/TrackState.hpp"
#include "Geant/track/TrackModifier.hpp"
#include "Geant/geometry/NavigationInterface.hpp"

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/Orb.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeom/volumes/Trapezoid.h"

#include "VecGeom/navigation/HybridNavigator2.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxLevelLocator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"
#include "VecGeom/navigation/VNavigator.h"

#include "Geant/magneticfield/FieldPropagationHandler.hpp"
#include "Geant/magneticfield/FieldPropagationHandler.hpp"
#include "Geant/core/LinearPropagationHandler.hpp"

#include "BasicCpuTransport/TrackManager.hpp"
#include "BasicCpuTransport/Types.hpp"
#include "BasicCpuTransport/ProxyDetectorConstruction.hpp"
#include "BasicCpuTransport/HepDetectorConstruction.hpp"

using namespace geantx;
using namespace vecgeom;

//===----------------------------------------------------------------------===//
// I doubt someone will see this but if you do, please replace with whatever
// is needed to get a random number generator from VecMath.
//
// And if this is Philippe looking, consider me warned about static thread_local
// being slow.
//
/*
// moved to Geant/processes/Process.hpp
inline double
get_rand()
{
    static auto _get_generator = []() {
        std::random_device rd;
        std::mt19937_64    gen(rd());
        return gen;
    };
    static thread_local auto _gen = _get_generator();
    return std::generate_canonical<double, 10>(_gen);
}
*/

//===----------------------------------------------------------------------===//

// VPlacedVolume *
void
initialize_geometry()
{
    UnplacedBox*       worldUnplaced = new UnplacedBox(10, 10, 10);
    UnplacedTrapezoid* trapUnplaced =
        new UnplacedTrapezoid(4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0);
    UnplacedBox* boxUnplaced = new UnplacedBox(2.5, 2.5, 2.5);
    UnplacedOrb* orbUnplaced = new UnplacedOrb(2.8);

    LogicalVolume* world = new LogicalVolume("world", worldUnplaced);
    LogicalVolume* trap  = new LogicalVolume("trap", trapUnplaced);
    LogicalVolume* box   = new LogicalVolume("box", boxUnplaced);
    LogicalVolume* orb   = new LogicalVolume("orb", orbUnplaced);

    Transformation3D* ident = new Transformation3D(0, 0, 0, 0, 0, 0);
    orb->PlaceDaughter("orb1", box, ident);
    trap->PlaceDaughter("box1", orb, ident);

    Transformation3D* placement1 = new Transformation3D(5, 5, 5, 0, 0, 0);
    Transformation3D* placement2 =
        new Transformation3D(-5, 5, 5, 0, 0, 0);  // 45,  0,  0);
    Transformation3D* placement3 = new Transformation3D(5, -5, 5, 0, 0, 0);  // 0, 45, 0);
    Transformation3D* placement4 =
        new Transformation3D(5, 5, -5, 0, 0, 0);  // 0,  0, 45);
    Transformation3D* placement5 =
        new Transformation3D(-5, -5, 5, 0, 0, 0);  // 45, 45,  0);
    Transformation3D* placement6 =
        new Transformation3D(-5, 5, -5, 0, 0, 0);  // 45,  0, 45);
    Transformation3D* placement7 =
        new Transformation3D(5, -5, -5, 0, 0, 0);  // 0, 45, 45);
    Transformation3D* placement8 =
        new Transformation3D(-5, -5, -5, 0, 0, 0);  // 45, 45, 45);

    world->PlaceDaughter("trap1", trap, placement1);
    world->PlaceDaughter("trap2", trap, placement2);
    world->PlaceDaughter("trap3", trap, placement3);
    world->PlaceDaughter("trap4", trap, placement4);
    world->PlaceDaughter("trap5", trap, placement5);
    world->PlaceDaughter("trap6", trap, placement6);
    world->PlaceDaughter("trap7", trap, placement7);
    world->PlaceDaughter("trap8", trap, placement8);

    VPlacedVolume* w = world->Place();
    GeoManager::Instance().SetWorld(w);
    GeoManager::Instance().CloseGeometry();
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void
ApplyAtRest(Track* track, size_t idx)
{
    using Apply_t = typename PhysicsProcessAtRest<ParticleType, ProcessTuple>::type;
    using Funct_t = std::function<void()>;
    TIMEMORY_BASIC_MARKER(toolset_t, "");

    geantx::Log(kInfo) << GEANT_HERE << "stepping track AtRest: " << *track;
    ///
    /// Reference: \file Geant/processes/ProcessConcepts.hpp
    ///
    ///     doit_idx:       index of smallest PIL
    ///     doit_value:     value of smallest PIL
    ///     doit_apply:     lambda that stores the "DoIt" for the smallest PIL
    ///
    intmax_t doit_idx   = -1;
    double   doit_value = std::numeric_limits<double>::max();
    Funct_t  doit_apply = [=]() {
        // geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
    };
    ///
    /// Calculate all of the AtRest PIL lengths for all the processes
    ///
    Apply<void>::unroll_indices<Apply_t>(track, &doit_idx, &doit_value, &doit_apply);
    ///
    /// Invoke the DoIt of smallest PIL
    ///
    doit_apply();
    ///
    /// Invoke all the AtRest processes that don't propose a PIL
    ///
    Apply<void>::apply<Apply_t>(track);
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void
ApplyAlongStep(Track* track, size_t idx)
{
    using Apply_t = typename PhysicsProcessAlongStep<ParticleType, ProcessTuple>::type;
    using Funct_t = std::function<void()>;
    TIMEMORY_BASIC_MARKER(toolset_t, "");

    geantx::Log(kInfo) << GEANT_HERE << "stepping track AlongStep: " << *track;
    ///
    /// Reference: \file Geant/processes/ProcessConcepts.hpp
    ///
    ///     doit_idx:       index of smallest PIL
    ///     doit_value:     value of smallest PIL
    ///     doit_apply:     lambda that stores the "DoIt" for the smallest PIL
    ///
    intmax_t doit_idx   = -1;
    double   doit_value = std::numeric_limits<double>::max();
    Funct_t  doit_apply = [=]() {
        // geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
    };
    ///
    /// Calculate all of the AlongStep PIL lengths for all the processes
    ///
    Apply<void>::unroll_indices<Apply_t>(track, &doit_idx, &doit_value, &doit_apply);
    ///
    /// Invoke the DoIt of smallest PIL
    ///
    doit_apply();
    ///
    /// Invoke all the AlongStep processes that don't propose a PIL
    ///
    Apply<void>::apply<Apply_t>(track);
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void
ApplyPostStep(Track* track, size_t idx)
{
    using Apply_t = typename PhysicsProcessPostStep<ParticleType, ProcessTuple>::type;
    using Funct_t = std::function<void()>;
    TIMEMORY_BASIC_MARKER(toolset_t, "");

    geantx::Log(kInfo) << GEANT_HERE << "stepping track PostStep: " << *track;
    ///
    /// Reference: \file Geant/processes/ProcessConcepts.hpp
    ///
    ///     doit_idx:       index of smallest PIL
    ///     doit_value:     value of smallest PIL
    ///     doit_apply:     lambda that stores the "DoIt" for the smallest PIL
    ///
    intmax_t doit_idx   = -1;
    double   doit_value = std::numeric_limits<double>::max();
    Funct_t  doit_apply = [=]() {
        geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
    };
    ///
    /// Calculate all of the PostStep PIL lengths for all the processes
    ///
    Apply<void>::unroll_indices<Apply_t>(track, &doit_idx, &doit_value, &doit_apply);
    ///
    /// Invoke the DoIt of smallest PIL
    ///
    doit_apply();
    ///
    /// Invoke all the PostStep processes that don't propose a PIL
    ///
    Apply<void>::apply<Apply_t>(track);
}


template <typename ParticleType, typename ProcessType, typename ParticleTypeProcesses>
struct AlongStepAtRestWrap
{
    template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<(_Proc::EnableAlongStepDoIt), int> = 0>
    AlongStepAtRestWrap(_Track* track)
    {
        using AtRestApply_t = typename PhysicsProcessAtRest<ParticleType, ParticleTypeProcesses>::type;

        ProcessType p(track);
        if (IsStopped(*track)) {
            if (IsAlive(*track))
                Apply<void>::apply<AtRestApply_t>(track);
            // instead 'return' we want to say 'abort/break-out-of loop'
            // for now we 'repeat pointlessly the test in AlongStep struct (in ProcessConcepts.hpp)
            // return;
        }
    }

    template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<!(_Proc::EnableAlongStepDoIt), int> = 0>
    AlongStepAtRestWrap(_Track* track)
    {
    }
};

template <typename ParticleType, typename AllProcessTypesTuple, typename AlongStepProcessTypes>
struct PhysicsProcessAlongStepAtRestWrap
{};

template <typename ParticleType, typename AllProcessTypesTuple, typename... AlongStepProcessTypes>
struct PhysicsProcessAlongStepAtRestWrap<ParticleType, AllProcessTypesTuple, std::tuple<AlongStepProcessTypes...>>
{
    using type = std::tuple<AlongStepAtRestWrap<ParticleType, AlongStepProcessTypes, AllProcessTypesTuple>...>;
};

template <typename ParticleType, typename ProcessType, typename ParticleTypeProcesses>
struct PostStepAtRestWrap
{
    template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<(_Proc::EnableAlwaysOnPostStepDoIt), int> = 0>
    PostStepAtRestWrap(_Track* track)
    {
        using AtRestApply_t = typename PhysicsProcessAtRest<ParticleType, ParticleTypeProcesses>::type;

        ProcessType p(track);
        if (IsStopped(*track)) {
            if (IsAlive(*track))
                Apply<void>::apply<AtRestApply_t>(track);
            // instead 'return' we want to say 'abort/break-out-of loop'
            // for now we 'repeat pointlessly the test in PostStep struct (in ProcessConcepts.hpp)
            // return;
        }
    }

    template <typename _Track, typename _Proc = ProcessType,
            std::enable_if_t<!(_Proc::EnableAlwaysOnPostStepDoIt), int> = 0>
    PostStepAtRestWrap(_Track* track)
    {
    }

};

template <typename ParticleType, typename AllProcessTypesTuple, typename PostStepProcessTypes>
struct PhysicsProcessPostStepAtRestWrap
{};

template <typename ParticleType, typename AllProcessTypesTuple, typename... PostStepProcessTypes>
struct PhysicsProcessPostStepAtRestWrap<ParticleType, AllProcessTypesTuple, std::tuple<PostStepProcessTypes...>>
{
    using type = std::tuple<PostStepAtRestWrap<ParticleType, PostStepProcessTypes, AllProcessTypesTuple>...>;
};

template <typename PropagationHandler>
VECCORE_ATT_HOST_DEVICE
GEANT_FORCE_INLINE
bool ReachedBoundary(TrackState &track, PropagationHandler &h, TaskData *td)
{
    const bool reachedBoundary = (track.fGeometryState.fSafety < 1.E-10) && !h.IsSameLocation(track, td);
    return reachedBoundary;
}

template <typename PropagationHandler>
VECCORE_ATT_HOST_DEVICE
GEANT_FORCE_INLINE
bool Propagate(TrackState &track, PropagationHandler &h, TaskData *td)
{
    if ( ! h.Propagate(track, td)) { // updates:  fPstep -= step;  fStep += step;  fSnext -= step
      return false;
    }
    ++(track.fHistoryState.fNsteps);
    while ( ! ReachedPhysicsLength(track) && ! ReachedBoundary(track, h, td)) {
        NavigationInterface::FindNextBoundary(track);
	if ( ! h.Propagate(track, td)) {
           return false;
	}
    }

    return true;
}


// Selector for Propagation engine.
// Note (to do later) it should also take the propagation handler as template argument to be customizable.
template <typename ParticleType, std::enable_if_t<ParticleType::kCharged, int> = 0>
bool Propagate(Track *track, TaskData *td)
{
    FieldPropagationHandler h;
    return Propagate(*track, h, td);
}

template <typename ParticleType, std::enable_if_t<!ParticleType::kCharged, int> = 0>
bool Propagate(Track *track, TaskData *td)
{
    LinearPropagationHandler h;
    return Propagate(*track, h, td);
}

//--------------------------------------------------------------------------------------//
// Inner part of one step for one track.
//
template <typename ParticleType, typename ParticleTypeProcesses, typename ProcessesFunction>
auto
InnerStep(Track *track,
          ProcessesFunction processes)
{
    using AtRestApply_t = typename PhysicsProcessAtRest<ParticleType, ParticleTypeProcesses>::type;
    using AlongStepApply_t = typename PhysicsProcessCombinedAlongStep<ParticleType, ParticleTypeProcesses>::type;
    using AlongStepAtRestApply_t = typename PhysicsProcessAlongStepAtRestWrap<ParticleType, ParticleTypeProcesses, AlongStepApply_t>::type;

    using PostStepApply_t = typename PhysicsProcessCombinedPostStep<ParticleType, ParticleTypeProcesses>::type;
    using PostStepAtRestApply_t = typename PhysicsProcessPostStepAtRestWrap<ParticleType, ParticleTypeProcesses, PostStepApply_t>::type;

    /// a. Integrate Equation of Motion
    /// b. if !alive return
    /// c. while did-not-reach-physics-length and did-not-cross-boundary
    ///       Find next Geometry boundary
    ///       Integrate Equation of Motion
    ///       if !alive return
    /// d. exec ProcessFunc

    geantx::Log(kInfo) << GEANT_HERE << "Inner step for: " << *track;

    if (!Propagate<ParticleType>(track, nullptr)) {
        return; // Particle is no longer alive
    }


    // Right now 'processes' is actually just the doit of the PostStep process if any,
    // See OneStep(Track *track) for a way to remove some of the loop and if at compile time
    // and then we could just have
    //     processes();
    // in the meantime be explicit:
    Apply<void>::apply<AlongStepAtRestApply_t>(track);

    // Currently there is only one selected PostStep and we are not yet creating the
    // interwine loop ('selector' + always-on) so we do not in the wrong order.
    if (!IsStopped(*track) && IsAlive(*track))
        processes();
    if (IsStopped(*track)) {
        if (IsAlive(*track))
           Apply<void>::apply<AtRestApply_t>(track);
        return;
    }
    Apply<void>::apply<PostStepAtRestApply_t>(track);

}

//--------------------------------------------------------------------------------------//
// One step for one track.
//
template <typename ParticleType, typename ParticleTypeProcesses>
auto
OneStep(Track *track)
{
    using PostStepApply_t = typename PhysicsProcessPostStep<ParticleType, ParticleTypeProcesses>::type;
    using AlongStepApply_t = typename PhysicsProcessAlongStep<ParticleType, ParticleTypeProcesses>::type;
    using Funct_t = std::function<void()>;

    TIMEMORY_BASIC_MARKER(toolset_t, "");

/*
    0. Reset track state for new 'step'
    1. Select process (Along or Post) which shortest ‘proposed physics interaction length’ -> [ Process, Type, Length, ProcessFunc ]
    2. Find next Geometry boundary
    3. MSC preparation
    4. InnerFunc(ProcessFunc)
    5. Sensitive Hit recording
    6. UserAction(s)
*/
    geantx::Log(kInfo) << GEANT_HERE << "One step for: " << *track;

    intmax_t doit_idx   = -1;
    Funct_t  doit_apply = [=]() {
        geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
    };
    ///
    /// Calculate all of the PIL lengths for all the PostStep processes and select one (or more)
    ///
    /// Note: G4 allow disabling of processes at run-time ....
    /// Ideally the returned function (doit_apply) would be a precompiled function containing:
    ///    For each along process
    ///       AlongStepDoIt
    ///       if stopped
    ///          if alive && has-alongStep-processes
    ///             exec AlongStep
    ///          return
    ///
    ///    For selected PostStep
    ///       PostStepDoIt
    ///      if stopped
    ///          if alive && has-post-step-processes
    ///             exec PostStep
    ///          return
    /// With as much pre-computed as possible (eg has-at-rest-processes)
    double   postPstep = std::numeric_limits<double>::max();
    Apply<void>::unroll_indices<PostStepApply_t>(track, &doit_idx, &postPstep, &doit_apply);


    ///
    /// Calculate all of the PIL lengths for all the AlongStep processes
    ///
    /// If one of the AlongStep process has the smallest PIL, doit_apply should be updated to be 'only':
    ///    For each along process
    ///       AlongStepDoIt
    ///       if stopped
    ///          if alive && has-alongStep-processes
    ///             exec AlongStep
    ///          return
    double   alongPstep = std::numeric_limits<double>::max();
    Apply<void>::unroll_indices<AlongStepApply_t>(track, &doit_idx, &alongPstep, &doit_apply);


    track->fPhysicsState.fPstep = vecCore::math::Min(alongPstep, postPstep);

    NavigationInterface::FindNextBoundary(*track);

    /// Apply multiple scattering if any.
    /// ...

    InnerStep<ParticleType, ParticleTypeProcesses>(track, doit_apply);

    /// Apply/do sensitive hit recording
    /// ....

    /// Apply/do user actions
    /// ....

    UpdateSwapPath(*track);

    /// Apply post step updates.
    //    ++track->fPhysicsState.fPstep;
    //track->fPhysicsState.fPstep = proposedPhysLength;

    //track->fStep = vecCore::math::Min(postPstep, alongPstep);  // GL: commented out, as it was resetting step at this point
}

//--------------------------------------------------------------------------------------//
// First example of a step
template <typename ParticleType, typename ParticleTypeProcesses>
auto
StepExample(Track *_track, size_t i)
{
    _track->fStatus = TrackStatus::Alive;
    ApplyAtRest<ParticleType, ParticleTypeProcesses>(_track, i);
    if(_track->fStatus != TrackStatus::Killed)
        ApplyAlongStep<ParticleType, ParticleTypeProcesses>(_track, i);
    if(_track->fStatus != TrackStatus::Killed)
        ApplyPostStep<ParticleType, ParticleTypeProcesses>(_track, i);
    ++_track->fPhysicsState.fPstep;
}

//--------------------------------------------------------------------------------------//
// converts all tracks to a vector (not-optimal) and then transport them in a loop
//
template <typename ParticlePhysics, typename... ParticleTypes,
          typename ParticleType          = typename ParticlePhysics::particle,
          typename ParticleTypeProcesses = typename ParticlePhysics::physics>
auto
DoStep(VariadicTrackManager<ParticleTypes...>* primary,
       VariadicTrackManager<ParticleTypes...>* secondary)
{
    TIMEMORY_BASIC_MARKER(toolset_t, "");

    // using AtRestProcesses =
    //     sort::sort<PhysicsProcessAtRestPriority, ParticleTypeProcesses>::type;

    //
    // here would be a memory transfer to GPU, if needed
    //
    while(!primary->Empty())
    {
        auto sz = primary->template Size<ParticleType>();
        for(size_t i = 0; i < sz; ++i)
        {
            auto* _track = primary->template PopTrack<ParticleType>(i);
            if(!_track) break;

            OneStep<ParticleType, ParticleTypeProcesses>(_track);

            ///
            /// Push the track back into primary
            ///
            secondary->template PushTrack<ParticleType>(_track, i);
        }
        // ... etc.
        std::swap(primary, secondary);
    }
}

//--------------------------------------------------------------------------------------//
// general implementation that launches kernels until all ready and secondary are
// finished
//
Track*
get_primary_particle(double Ekin)
{
    TIMEMORY_BASIC_MARKER(toolset_t, "");
    Track* _track = new Track;

    //TrackModifier<CpuElectron> trk(_track);

    //_track->fPhysicsState.fEkin  = Ekin;
    UpdateEkin(*_track, Ekin);
    _track->fDir  = { get_rand(), get_rand(), get_rand() };
    _track->fPos  = { get_rand(), get_rand(), get_rand() };
    _track->fDir.Normalize();

    // set volume path
    vecgeom::GlobalLocator::LocateGlobalPoint(vecgeom::GeoManager::Instance().GetWorld(), _track->fPos,
					      *_track->fGeometryState.fPath, true);

    auto top = _track->fGeometryState.fPath->Top();
    auto *vol = (top) ? top->GetLogicalVolume() : nullptr;
    _track->fGeometryState.fVolume = vol;
    _track->fMaterialState.fMaterial = ((Material_t *)vol->GetMaterialPtr());
    //TODO: Get material index correctly
    _track->fMaterialState.fMaterialId = _track->fMaterialState.fMaterial->GetIndex() - 1;
    std::cout<<"* getPrimPart: pos="<< _track->fPos <<", dir="<< _track->fDir
	     <<", volume=<"<< _track->fGeometryState.fPath->Top()->GetLabel()
	     <<"> : "<< *_track <<"\n";

    return _track;
}

//===----------------------------------------------------------------------===//

int
main(int argc, char** argv)
{
    tim::settings::precision() = 6;
    tim::settings::width()     = 12;
    tim::timemory_init(argc, argv);

    TIMEMORY_BLANK_MARKER(toolset_t, argv[0]);

    /*
    if(tim::get_env<bool>("TRANSPORT_GEOM", false))
    {
        initialize_geometry();
        // basic geometry checks
        if(GeoManager::Instance().GetWorld())
        {
            const auto* logWorld = GeoManager::Instance().GetWorld()->GetLogicalVolume();
            if(logWorld)
            {
                // print detector information
                logWorld->PrintContent();
                std::cout << "\n # placed volumes: " << logWorld->GetNTotal() << "\n";
            }
        }
    }
    */

    // Create and configure run manager
    geantx::RunManager *runMgr = NULL;

    // Create detector construction
    geantx::UserDetectorConstruction *det = new userapplication::ProxyDetectorConstruction(runMgr);

    geantx::vector_t<geantx::Volume_t const *> volumes;
    int numVolumes = 0;
    if (det) {
      det->CreateMaterials();
      det->CreateGeometry();
      numVolumes = det->SetupGeometry(volumes);
    }

    std::cout << " Number of the maximum volume depth = " << numVolumes << std::endl;

    // initialize navigation
    det->InitNavigators();

    VariadicTrackManager<CpuGamma, CpuElectron, GpuGamma, GpuElectron> primary;
    VariadicTrackManager<CpuGamma, CpuElectron, GpuGamma, GpuElectron> secondary;

    /*
    //
    // This will eventually provide a re-ordering for the sequence of which the
    // processes are applied. So one can do something like:
    //
    //      template <>
    //      struct PhysicsProcessPostStepPriority<Transportation>
    //      : std::integral_constant<int, -100>
    //      {};
    //
    //      template <>
    //      struct PhysicsProcessPostStepPriority<ProxyStepLimiter>
    //      : std::integral_constant<int, 100>
    //      {};
    //
    //  so that for the PostStep stage, Transportation is prioritized ahead of
    //  other processes and ProxyStepLimiter is, in general, applied after
    //  most other processes
    //

    using PhysList_A = std::tuple<ProxyTrackLimiter>;
    using PhysList_B =
        InsertSorted<PhysList_A, ProxyScattering, SortPhysicsProcessPostStep>;
    using PhysList_C =
        InsertSorted<PhysList_B, ProxyStepLimiter, SortPhysicsProcessPostStep>;
    using PhysList_D =
        InsertSorted<PhysList_C, Transportation, SortPhysicsProcessPostStep>;

    std::cout << "Phys List A : " << tim::demangle(typeid(PhysList_A).name())
              << std::endl;
    std::cout << "Phys List B : " << tim::demangle(typeid(PhysList_B).name())
              << std::endl;
    std::cout << "Phys List C : " << tim::demangle(typeid(PhysList_C).name())
              << std::endl;
    std::cout << "Phys List D : " << tim::demangle(typeid(PhysList_D).name())
              << std::endl;
    */

    // prepare primary tracks - TODO: use a particle gun
    double energy = 10. * geantx::clhep::GeV;

    printf("* CpuGammas\n");
    primary.PushTrack<CpuGamma>(get_primary_particle(energy));
    primary.PushTrack<CpuGamma>(get_primary_particle(energy));
    primary.PushTrack<CpuGamma>(get_primary_particle(energy));
    primary.PushTrack<CpuGamma>(get_primary_particle(energy));
    primary.PushTrack<CpuGamma>(get_primary_particle(energy));
    primary.PushTrack<CpuGamma>(get_primary_particle(energy));

    printf("\n");
    primary.PushTrack<GpuGamma>(get_primary_particle(energy));

    printf("* CpuElectrons\n");
    primary.PushTrack<CpuElectron>(get_primary_particle(energy));
    primary.PushTrack<CpuElectron>(get_primary_particle(energy));

    printf("* GpuElectrons\n");
    primary.PushTrack<GpuElectron>(get_primary_particle(energy));

    printf("* GpuGammas\n");
    primary.PushTrack<GpuGamma>(get_primary_particle(energy));

    //stepping

    printf("** Processing CpuGammas...\n");
    DoStep<CpuGammaPhysics>(&primary, &secondary);

    printf("** Processing CpuElectrons\n");
    DoStep<CpuElectronPhysics>(&primary, &secondary);

    printf("** Processing GpuGammas\n");
    DoStep<GpuGammaPhysics>(&primary, &secondary);

    printf("** Processing GpuElectrons\n");
    DoStep<GpuElectronPhysics>(&primary, &secondary);
}

//===----------------------------------------------------------------------===//
