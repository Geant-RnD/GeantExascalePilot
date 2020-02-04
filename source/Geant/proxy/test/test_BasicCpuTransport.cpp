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
#include "Geant/core/SystemOfUnits.hpp"
#include "Geant/core/Tasking.hpp"
#include "Geant/geometry/UserDetectorConstruction.hpp"
#include "Geant/track/TrackState.hpp"

#include "management/GeoManager.h"
#include "volumes/Box.h"
#include "volumes/LogicalVolume.h"
#include "volumes/Orb.h"
#include "volumes/PlacedVolume.h"
#include "volumes/Trapezoid.h"

#include "navigation/HybridNavigator2.h"
#include "navigation/NewSimpleNavigator.h"
#include "navigation/SimpleABBoxLevelLocator.h"
#include "navigation/SimpleABBoxNavigator.h"
#include "navigation/VNavigator.h"

#include "BasicCpuTransport/ProxyDetectorConstruction.hpp"
#include "BasicCpuTransport/TrackManager.hpp"
#include "BasicCpuTransport/Types.hpp"

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
void initialize_geometry()
{
  UnplacedBox *worldUnplaced = new UnplacedBox(10, 10, 10);
  UnplacedTrapezoid *trapUnplaced =
      new UnplacedTrapezoid(4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0);
  UnplacedBox *boxUnplaced = new UnplacedBox(2.5, 2.5, 2.5);
  UnplacedOrb *orbUnplaced = new UnplacedOrb(2.8);

  LogicalVolume *world = new LogicalVolume("world", worldUnplaced);
  LogicalVolume *trap  = new LogicalVolume("trap", trapUnplaced);
  LogicalVolume *box   = new LogicalVolume("box", boxUnplaced);
  LogicalVolume *orb   = new LogicalVolume("orb", orbUnplaced);

  Transformation3D *ident = new Transformation3D(0, 0, 0, 0, 0, 0);
  orb->PlaceDaughter("orb1", box, ident);
  trap->PlaceDaughter("box1", orb, ident);

  Transformation3D *placement1 = new Transformation3D(5, 5, 5, 0, 0, 0);
  Transformation3D *placement2 = new Transformation3D(-5, 5, 5, 0, 0, 0);  // 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D(5, -5, 5, 0, 0, 0);  // 0, 45, 0);
  Transformation3D *placement4 = new Transformation3D(5, 5, -5, 0, 0, 0);  // 0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5, 5, 0, 0, 0); // 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-5, 5, -5, 0, 0, 0); // 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D(5, -5, -5, 0, 0, 0); // 0, 45, 45);
  Transformation3D *placement8 =
      new Transformation3D(-5, -5, -5, 0, 0, 0); // 45, 45, 45);

  world->PlaceDaughter("trap1", trap, placement1);
  world->PlaceDaughter("trap2", trap, placement2);
  world->PlaceDaughter("trap3", trap, placement3);
  world->PlaceDaughter("trap4", trap, placement4);
  world->PlaceDaughter("trap5", trap, placement5);
  world->PlaceDaughter("trap6", trap, placement6);
  world->PlaceDaughter("trap7", trap, placement7);
  world->PlaceDaughter("trap8", trap, placement8);

  VPlacedVolume *w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void ApplyAtRest(Track *track, size_t idx)
{
  using Apply_t = typename PhysicsProcessAtRest<ParticleType, ProcessTuple>::type;
  using Apply_v = Apply<void>;
  using Func_t  = std::function<void()>;

  TIMEMORY_BASIC_MARKER(toolset_t, "");

  geantx::Log(kInfo) << GEANT_HERE << "stepping track AtRest: " << *track;
  ///
  /// Reference: \file Geant/processes/ProcessConcepts.hpp
  ///
  ///     doit_idx:       index of smallest PIL
  ///     doit_value:     value of smallest PIL
  ///     doit_apply:     lambda that stores the "DoIt" for the smallest PIL
  ///
  intmax_t doit_idx = -1;
  double doit_value = std::numeric_limits<double>::max();
  Func_t doit_apply = [=]() {
    // geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
  };
  ///
  /// Calculate all of the AtRest PIL lengths for all the processes
  ///
  Apply_v::unroll_indices<Apply_t>(track, &doit_idx, &doit_value, &doit_apply);
  ///
  /// Invoke the DoIt of smallest PIL
  ///
  doit_apply();
  ///
  /// Invoke all the AtRest processes that don't propose a PIL
  ///
  Apply_v::apply<Apply_t>(track);
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void ApplyAlongStep(Track *track, size_t idx)
{
  using Apply_t = typename PhysicsProcessAlongStep<ParticleType, ProcessTuple>::type;
  using Apply_v = Apply<void>;
  using Func_t  = std::function<void()>;

  TIMEMORY_BASIC_MARKER(toolset_t, "");

  geantx::Log(kInfo) << GEANT_HERE << "stepping track AlongStep: " << *track;
  ///
  /// Reference: \file Geant/processes/ProcessConcepts.hpp
  ///
  ///     doit_idx:       index of smallest PIL
  ///     doit_value:     value of smallest PIL
  ///     doit_apply:     lambda that stores the "DoIt" for the smallest PIL
  ///
  intmax_t doit_idx = -1;
  double doit_value = std::numeric_limits<double>::max();
  Func_t doit_apply = [=]() {
    // geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
  };
  ///
  /// Calculate all of the AlongStep PIL lengths for all the processes
  ///
  Apply_v::unroll_indices<Apply_t>(track, &doit_idx, &doit_value, &doit_apply);
  ///
  /// Invoke the DoIt of smallest PIL
  ///
  doit_apply();
  ///
  /// Invoke all the AlongStep processes that don't propose a PIL
  ///
  Apply_v::apply<Apply_t>(track);
}

//--------------------------------------------------------------------------------------//

template <typename ParticleType, typename ProcessTuple>
void ApplyPostStep(Track *track, size_t idx)
{
  using Apply_t = typename PhysicsProcessPostStep<ParticleType, ProcessTuple>::type;
  using Apply_v = Apply<void>;
  using Func_t  = std::function<void()>;

  TIMEMORY_BASIC_MARKER(toolset_t, "");

  geantx::Log(kInfo) << GEANT_HERE << "stepping track PostStep: " << *track;
  ///
  /// Reference: \file Geant/processes/ProcessConcepts.hpp
  ///
  ///     doit_idx:       index of smallest PIL
  ///     doit_value:     value of smallest PIL
  ///     doit_apply:     lambda that stores the "DoIt" for the smallest PIL
  ///
  intmax_t doit_idx = -1;
  double doit_value = std::numeric_limits<double>::max();
  Func_t doit_apply = [=]() {
    geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
  };
  ///
  /// Calculate all of the PostStep PIL lengths for all the processes
  ///
  Apply_v::unroll_indices<Apply_t>(track, &doit_idx, &doit_value, &doit_apply);
  ///
  /// Invoke the DoIt of smallest PIL
  ///
  doit_apply();
  ///
  /// Invoke all the PostStep processes that don't propose a PIL
  ///
  Apply_v::apply<Apply_t>(track);
}

//--------------------------------------------------------------------------------------//
// One step for one track.
//
template <typename ParticleType, typename ParticleTypeProcesses>
auto OneStep(Track *track)
{
  using Func_t  = std::function<void()>;
  using Apply_v = Apply<void>;

  using AtRest_t    = PhysicsProcessAtRest<ParticleType, ParticleTypeProcesses>;
  using AlongStep_t = PhysicsProcessAlongStep<ParticleType, ParticleTypeProcesses>;
  using PostStep_t  = PhysicsProcessPostStep<ParticleType, ParticleTypeProcesses>;

  using AtRestApply_t    = typename AtRest_t::type;
  using PostStepApply_t  = typename PostStep_t::type;
  using AlongStepApply_t = typename AlongStep_t::type;

  // using AlongStepAtRestApply_t =
  //    typename AlongStep_t::template ApplyAtRest<Track, AtRestApply_t>;
  // using PostStepAtRestApply_t =
  //    typename PostStep_t::template ApplyAtRest<Track, AtRestApply_t>;

  TIMEMORY_BASIC_MARKER(toolset_t, "");

  /*
      0. Reset track state for new 'step'
      1. Select process (Along or Post) which shortest proposed physics interaction
     length -> [ Process, Type, Length, ProcessFunc ]
      2. Find next Geometry boundary
      3. MSC preparation
      4. InnerFunc(ProcessFunc)
      5. Sensitive Hit recording
      6. UserAction(s)
  */

  geantx::Log(kInfo) << GEANT_HERE << "One step for: " << *track;

  if (IsStopped(*track)) {
    //
    //  excerpt from G4SteppingManager::Stepping()
    //
    //      if( fTrack->GetTrackStatus() == fStopButAlive )
    //      {
    //          if( MAXofAtRestLoops>0 ) { ... }
    //          // Make sure the track is killed
    //          fTrack->SetTrackStatus( fStopAndKill );
    //      }
    //      else
    //      { ... }  <-- AlongStep and PostStep
    //
    //  In other words, if a particle is stopped, there is no invocation
    //  or AlongStep or PostStep after AtRest and there are no AtRest calls
    //  after AlongStep or PostStep
    //
    //  Ideally, I think we should push all stopped particles to the
    //  generic queue when they are created and handling them at all
    //  on the GPU since they require no transport steps
    //
    geantx::Log(kInfo) << GEANT_HERE << "Stopped track: " << *track;
    TIMEMORY_BASIC_MARKER(toolset_t, "AtRest track");
    Apply_v::apply<AtRestApply_t>(track);
    track->fStatus = TrackStatus::Killed;
  } else {
    intmax_t doit_idx         = -1;
    double proposedPhysLength = std::numeric_limits<double>::max();
    Func_t doit_apply         = [=]() {
      geantx::Log(kInfo) << GEANT_HERE << "no process selected: " << *track;
    };

    ///
    /// Calculate all of the PIL lengths for all the PostStep processes and select one
    /// (or more)
    ///
    /// Note: G4 allow disabling of processes at run-time ....
    /// Ideally the returned function (doit_apply) would be a precompiled function
    /// containing:
    ///    For each along process
    ///       AlongStepDoIt
    ///       if stopped                                (INVALID -- J. MADSEN)
    ///          if alive && has-at-rest-processes      (INVALID -- J. MADSEN)
    ///             exec AtRest                         (INVALID -- J. MADSEN)
    ///          return                                 (INVALID -- J. MADSEN)
    ///
    ///    For selected PostStep
    ///       PostStepDoIt
    ///      if stopped                                 (INVALID -- J. MADSEN)
    ///          if alive && has-at-rest-processes      (INVALID -- J. MADSEN)
    ///             exec AtRest                         (INVALID -- J. MADSEN)
    ///          return                                 (INVALID -- J. MADSEN)
    ///
    /// With as much pre-computed as possible (eg has-at-rest-processes)
    Apply_v::unroll_indices<PostStepApply_t>(track, &doit_idx, &proposedPhysLength,
                                             &doit_apply);

    ///
    /// Calculate all of the PIL lengths for all the AlongStep processes
    ///
    /// If one of the AlongStep process has the smallest PIL, doit_apply should be
    /// updated to be 'only':
    ///    For each along process
    ///       AlongStepDoIt
    ///       if stopped                                (INVALID -- J. MADSEN)
    ///          if alive && has-at-rest-processes      (INVALID -- J. MADSEN)
    ///             exec AtRest                         (INVALID -- J. MADSEN)
    ///          return                                 (INVALID -- J. MADSEN)
    ///
    Apply_v::unroll_indices<AlongStepApply_t>(track, &doit_idx, &proposedPhysLength,
                                              &doit_apply);

    /// FindNextBoundary(track);

    /// Apply multiple scaterring if any.
    /// ...

    /// a. Integrate Equation of Motion
    /// b. if !alive return
    /// c. while did-not-reach-physics-length and did-not-cross-boundary
    ///       Find next Geometry boundary
    ///       Integrate Equation of Motion
    ///       if !alive return
    /// d. exec ProcessFunc

    geantx::Log(kInfo) << GEANT_HERE << "Inner step for: " << *track;
    /*
        if (!Propagate<ParticleType>(track)) {
            return; // Particle is no longer alive
        }
        while( !ReachedPhysicsLength(track) && !ReachedBoundary(track) ) {
            FindNextBoundary(track);
            if (!Propagate<ParticleType>(track)) {
                return; // Particle is no longer alive
            }
        }
    */

    // Right now 'processes' is actually just the doit of the PostStep process if any,
    // See OneStep(Track *track) for a way to remove some of the loop and if at
    // compile time and then we could just have
    //     processes();
    // in the meantime be explicit:

    Apply_v::apply<AlongStepApply_t>(track);
    // Apply_v::apply<AlongStepAtRestApply_t>(track);

    // Currently there is only one selected PostStep and we are not yet creating the
    // interwine loop ('selector' + always-on) so we do not in the wrong order.

    // if(!IsStopped(*track) && IsAlive(*track)) doit_apply();
    if (IsAlive(*track)) doit_apply();

    /*
    if(IsStopped(*track))
    {
        if(IsAlive(*track)) Apply_v::apply<AtRestApply_t>(track);
        return;
    }
    */

    Apply_v::apply<PostStepApply_t>(track);
    // Apply_v::apply<PostStepAtRestApply_t>(track);
  }

  /// Apply/do sensitive hit recording
  /// ....

  /// Apply/do user actions
  /// ....

  /// Apply post step updates.
  ++track->fPhysicsState.fPstep;
}

//--------------------------------------------------------------------------------------//
// converts all tracks to a vector (not-optimal) and then transport them in a loop
//
template <typename ParticlePhysics, typename... ParticleTypes,
          typename ParticleType          = typename ParticlePhysics::particle,
          typename ParticleTypeProcesses = typename ParticlePhysics::physics>
auto DoStep(VariadicTrackManager<ParticleTypes...> *primary,
            VariadicTrackManager<ParticleTypes...> *secondary)
{
  TIMEMORY_BASIC_MARKER(toolset_t, "");

  // using AtRestProcesses =
  //     sort::sort<PhysicsProcessAtRestPriority, ParticleTypeProcesses>::type;

  //
  // here would be a memory transfer to GPU, if needed
  //
  while (!primary->Empty()) {
    auto sz = primary->template Size<ParticleType>();
    for (size_t i = 0; i < sz; ++i) {
      auto *_track = primary->template PopTrack<ParticleType>(i);
      if (!_track) break;

      // _track->transfer_to(geantx::device::gpu{});
      // auto _device_track = _track->device_ptr();
      // _track = _device_track.get();

      geantx::Log(kInfo) << "";
      OneStep<ParticleType, ParticleTypeProcesses>(_track);
      geantx::Log(kInfo) << "";

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
Track *get_primary_particle(VolumePath_t *startpath, double Ekin)
{
  TIMEMORY_BASIC_MARKER(toolset_t, "");
  Track *_track               = new Track;
  _track->fPhysicsState.fEkin = Ekin;
  _track->fDir                = {get_rand(), get_rand(), get_rand()};
  _track->fPos                = {get_rand(), get_rand(), get_rand()};
  _track->fDir.Normalize();

  _track->fGeometryState.fPath     = startpath;
  _track->fGeometryState.fNextpath = startpath;
  auto top                         = startpath->Top();
  auto *vol                        = (top) ? top->GetLogicalVolume() : nullptr;
  _track->fGeometryState.fVolume   = vol;
  _track->fMaterialState.fMaterial = ((Material_t *)vol->GetMaterialPtr());

  return _track;
}

//===----------------------------------------------------------------------===//

template <typename _Particle, typename _Physics, typename _Manager>
void primary_generator_action(VolumePath_t *startpath, double Ekin, int n = 1)
{
  if (std::is_same<_Particle, GpuElectron>::value) Ekin = 0.0;
  _Manager primary(TrackManager::Instance<0>());
  _Manager secondary(TrackManager::Instance<1>());
  for (int i = 0; i < n; ++i)
    primary.template PushTrack<_Particle>(get_primary_particle(startpath, Ekin));
  DoStep<_Physics>(&primary, &secondary);
}

//===----------------------------------------------------------------------===//

int main(int argc, char **argv)
{
  tim::settings::precision() = 6;
  tim::settings::width()     = 12;
  tim::timemory_init(argc, argv);

  int nthread = 0;
  int npart   = 1;
  int ntype   = 0;

  if (argc > 1) nthread = atoi(argv[1]);
  if (argc > 2) npart = atoi(argv[2]);
  if (argc > 3) ntype = atoi(argv[3]);

  using PhysList_normal = std::tuple<Transportation, ProxyStepLimiter, ProxyScattering,
                                     ProxySecondaryGenerator, ProxyTrackLimiter>;

  using PhysList_sorted = Sort<PhysicsProcessPostStepPriority, PhysList_normal>;

  std::cout << "Phys List (normal) : " << tim::demangle<PhysList_normal>() << std::endl;
  std::cout << "Phys List (sorted) : " << tim::demangle<PhysList_sorted>() << std::endl;

  TIMEMORY_BLANK_MARKER(toolset_t, argv[0]);

  // Create and configure run manager
  geantx::RunManager *runMgr = NULL;

  // Create CMS detector construction
  userapplication::ProxyDetectorConstruction *det =
      new userapplication::ProxyDetectorConstruction(runMgr);
  geantx::vector_t<geantx::Volume_t const *> volumes;

  int numVolumes = 0;
  if (det) {
    det->CreateMaterials();
    det->CreateGeometry();
    numVolumes = det->SetupGeometry(volumes);
  }
  det->DetectorInfo();
  std::cout << " Number of the maximum volume depth = " << numVolumes << std::endl;

  // initialize negivation
  det->InitNavigators();

  using TrackManager_t =
      VariadicTrackManager<CpuGamma, CpuElectron, GpuGamma, GpuElectron>;

  // at the beginning of an event/tracking - initialize the navigation path
  int maxDepth = vecgeom::GeoManager::Instance().getMaxDepth();
  vecgeom::Vector3D<double> vertex(0., 0., 0.);
  geantx::VolumePath_t *startpath = geantx::VolumePath_t::MakeInstance(maxDepth);
  startpath->Clear();
  vecgeom::GlobalLocator::LocateGlobalPoint(vecgeom::GeoManager::Instance().GetWorld(),
                                            vertex, *startpath, true);

  // prepare primary tracks - TODO: use a particle gun
  double energy = 10. * geantx::units::GeV;

  // stepping
  auto &cpu_tasking = geantx::tasking::cpu_run_manager();
  geantx::tasking::init_run_manager(cpu_tasking, nthread);
  geantx::tasking::TaskGroup<void> cpu_tg(cpu_tasking->GetThreadPool());

  auto cpu_gamma_exec = [&]() {
    primary_generator_action<CpuGamma, CpuGammaPhysics, TrackManager_t>(startpath, energy,
                                                                        npart);
  };

  auto cpu_electron_exec = [&]() {
    primary_generator_action<CpuElectron, CpuElectronPhysics, TrackManager_t>(
        startpath, energy, npart);
  };

  auto gpu_gamma_exec = [&]() {
    primary_generator_action<GpuGamma, GpuGammaPhysics, TrackManager_t>(startpath, energy,
                                                                        npart);
  };

  auto gpu_electron_exec = [&]() {
    primary_generator_action<GpuElectron, GpuElectronPhysics, TrackManager_t>(
        startpath, energy, npart);
  };

  std::vector<std::function<void()>> work = {cpu_gamma_exec, cpu_electron_exec,
                                             gpu_gamma_exec, gpu_electron_exec};

  int ncount = 0;
  for (auto &itr : work) {
    if (ntype > 0 && ++ncount > ntype) break;
    if (nthread == 0)
      itr();
    else
      cpu_tg.exec(itr);
  }
  printf("\n");
  cpu_tg.join();

  printf("Finalizing timemory...\n");
  tim::timemory_finalize();

  printf("\n");
}

//===----------------------------------------------------------------------===//
