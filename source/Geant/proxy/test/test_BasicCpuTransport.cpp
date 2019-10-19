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
#include "Geant/core/SystemOfUnits.hpp"
#include "Geant/geometry/RunManager.hpp"
#include "Geant/geometry/UserDetectorConstruction.hpp"
#include "Geant/track/TrackState.hpp"

#include "management/GeoManager.h"
#include "volumes/Box.h"
#include "volumes/LogicalVolume.h"
#include "volumes/Orb.h"
#include "volumes/PlacedVolume.h"
#include "volumes/Trapezoid.h"

#include "BasicCpuTransport/TrackManager.hpp"
#include "BasicCpuTransport/Types.hpp"

using namespace geantx;
using namespace vecgeom;

#include <random>

using namespace tim::component;
using toolset_t = tim::auto_tuple<real_clock, peak_rss>;
// using toolset_t =
//    tim::auto_tuple<real_clock, system_clock, user_clock, cpu_util, peak_rss>;
// using toolset_t = tim::auto_timer;

//===----------------------------------------------------------------------===//
// I doubt someone will see this but if you do, please replace with whatever
// is needed to get a random number generator from VecMath.
//
// And if this is Philippe looking, consider me warned about static thread_local
// being slow.
//
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

//===----------------------------------------------------------------------===//

// VPlacedVolume *
void
initialize_geometry()
{
    /*
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
    // return w;
    */
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void
ApplyAtRest(TrackCaster<ParticleType>** tracks, intmax_t N)
{
    using Apply_t = typename PhysicsProcessAtRest<ParticleType, ProcessTuple>::type;
    using Funct_t = std::function<void()>;
    TIMEMORY_BASIC_MARKER(toolset_t, "");

    for(int i = 0; i < N; ++i)
    {
        TrackState* track = static_cast<TrackState*>(tracks[i]);
        geantx::Log(kInfo) << GEANT_HERE << "stepping track: " << *track;
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
    // ... etc.
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void
ApplyAlongStep(TrackCaster<ParticleType>** tracks, intmax_t N)
{
    using Apply_t = typename PhysicsProcessAlongStep<ParticleType, ProcessTuple>::type;
    using Funct_t = std::function<void()>;
    TIMEMORY_BASIC_MARKER(toolset_t, "");

    for(int i = 0; i < N; ++i)
    {
        TrackState* track = static_cast<TrackState*>(tracks[i]);
        geantx::Log(kInfo) << GEANT_HERE << "stepping track: " << *track;
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
    // ... etc.
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void
ApplyPostStep(TrackCaster<ParticleType>** tracks, intmax_t N)
{
    using Apply_t = typename PhysicsProcessPostStep<ParticleType, ProcessTuple>::type;
    using Funct_t = std::function<void()>;
    TIMEMORY_BASIC_MARKER(toolset_t, "");

    for(int i = 0; i < N; ++i)
    {
        TrackState* track = static_cast<TrackState*>(tracks[i]);
        geantx::Log(kInfo) << GEANT_HERE << "stepping track: " << *track;
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
    // ... etc.
}

//--------------------------------------------------------------------------------------//
// converts all tracks to a vector (not-optimal) and then transport them in a loop
//
template <typename ParticlePhysics, typename... ParticleTypes,
          typename ParticleType          = typename ParticlePhysics::particle,
          typename ParticleTypeProcesses = typename ParticlePhysics::physics>
auto
DoStep(VariadicTrackManager<ParticleTypes...>* track_manager)
{
    using VarManager_t = VariadicTrackManager<ParticleTypes...>;
    using Track_t =
        decltype(std::declval<VarManager_t>().template PopTrack<ParticleType>());

    TIMEMORY_BASIC_MARKER(toolset_t, "");

    //
    //  put all data into array
    //
    std::vector<Track_t> _vec;
    while(true)
    {
        auto _track = track_manager->template PopTrack<ParticleType>();
        if(_track == nullptr) break;
        geantx::Log(kInfo) << GEANT_HERE << "stepping track: " << *_track;
        _vec.emplace_back(_track);
    }

    //
    // here would be a memory transfer to GPU, if needed
    //
    ApplyAtRest<ParticleType, ParticleTypeProcesses>(_vec.data(), _vec.size());
    ApplyAlongStep<ParticleType, ParticleTypeProcesses>(_vec.data(), _vec.size());
    ApplyPostStep<ParticleType, ParticleTypeProcesses>(_vec.data(), _vec.size());
}

//--------------------------------------------------------------------------------------//
// general implementation that launches kernels until all ready and secondary are
// finished
//
TrackState*
get_primary_particle()
{
    TIMEMORY_BASIC_MARKER(toolset_t, "");
    TrackState* _track = new TrackState;
    _track->fDir       = { get_rand(), get_rand(), get_rand() };
    _track->fPos       = { get_rand(), get_rand(), get_rand() };
    _track->fDir.Normalize();
    return _track;
}

//===----------------------------------------------------------------------===//

int
main(int argc, char** argv)
{
    TIMEMORY_BLANK_MARKER(toolset_t, argv[0]);
    initialize_geometry();

    // basic geometry checks
    /*if(GeoManager::Instance().GetWorld())
    {
        const auto* logWorld = GeoManager::Instance().GetWorld()->GetLogicalVolume();
        if(logWorld)
        {
            // print detector information
            logWorld->PrintContent();
            std::cout << "\n # placed volumes: " << logWorld->GetNTotal() << "\n";
        }
    }*/

    VariadicTrackManager<CpuGamma, CpuElectron, GpuGamma, GpuElectron> manager;

    printf("\n");
    manager.PushTrack(get_primary_particle());

    printf("\n");
    manager.PushTrack<CpuGamma>(get_primary_particle());
    manager.PushTrack<CpuGamma>(get_primary_particle());

    printf("\n");
    manager.PushTrack<GpuGamma>(get_primary_particle());

    printf("\n");
    manager.PushTrack<CpuElectron>(get_primary_particle());

    printf("\n");
    manager.PushTrack<GpuElectron>(get_primary_particle());

    printf("\n");
    DoStep<CpuGammaPhysics>(&manager);

    printf("\n");
    DoStep<CpuElectronPhysics>(&manager);

    printf("\n");
    DoStep<GpuGammaPhysics>(&manager);

    printf("\n");
    DoStep<GpuElectronPhysics>(&manager);
}

//===----------------------------------------------------------------------===//
