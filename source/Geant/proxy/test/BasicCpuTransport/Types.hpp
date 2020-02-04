//
//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===--------------------------------------------------------------------------------===//
//
/**
 * @file
 * @brief
 */
//===--------------------------------------------------------------------------------===//
//

#pragma once

#include "Geant/processes/ProcessConcepts.hpp"
#include "Geant/processes/Transportation.hpp"

#include "Geant/proxy/ProxyParticles.hpp"
#include "Geant/proxy/ProxyScattering.hpp"
#include "Geant/proxy/ProxySecondaryGenerator.hpp"
#include "Geant/proxy/ProxyStepLimiter.hpp"
#include "Geant/proxy/ProxyTrackLimiter.hpp"

// #include "Geant/proxy/ProxyBremsstrahlung.hpp"
// #include "Geant/proxy/ProxyCompton.hpp"
// #include "Geant/proxy/ProxyIonization.hpp"

using namespace geantx;

//===--------------------------------------------------------------------------------===//
//              Sorting the Type-Traits
//===--------------------------------------------------------------------------------===//

template <bool _Val, typename _Lhs, typename _Rhs>
using conditional_t = typename std::conditional<_Val, _Lhs, _Rhs>::type;

//--------------------------------------------------------------------------------------//

template <template <typename> class _Prio, typename _Beg, typename _Tp, typename _End>
struct SortT;

//--------------------------------------------------------------------------------------//

template <template <typename> class _Prio, typename _Tuple>
using Sort = typename SortT<_Prio, _Tuple, std::tuple<>, std::tuple<>>::type;

//--------------------------------------------------------------------------------------//
//  Initiate recursion (zeroth sort operation)
//
template <template <typename> class _Prio, typename _In, typename... _InT>
struct SortT<_Prio, std::tuple<_In, _InT...>, std::tuple<>, std::tuple<>> {
  using type =
      typename SortT<_Prio, std::tuple<_InT...>, std::tuple<>, std::tuple<_In>>::type;
};

//--------------------------------------------------------------------------------------//
//  Initiate recursion (zeroth sort operation)
//
template <template <typename> class _Prio, typename _In, typename... _InT>
struct SortT<_Prio, std::tuple<std::tuple<_In, _InT...>>, std::tuple<>, std::tuple<>> {
  using type =
      typename SortT<_Prio, std::tuple<_InT...>, std::tuple<>, std::tuple<_In>>::type;
};

//--------------------------------------------------------------------------------------//
//  Terminate recursion (last sort operation)
//
template <template <typename> class _Prio, typename... _BegT, typename... _EndT>
struct SortT<_Prio, std::tuple<>, std::tuple<_BegT...>, std::tuple<_EndT...>> {
  using type = std::tuple<_BegT..., _EndT...>;
};

//--------------------------------------------------------------------------------------//
//  If no current end, transfer begin to end ()
//
template <template <typename> class _Prio, typename _In, typename... _InT,
          typename... _BegT>
struct SortT<_Prio, std::tuple<_In, _InT...>, std::tuple<_BegT...>, std::tuple<>> {
  using type = typename SortT<_Prio, std::tuple<_In, _InT...>, std::tuple<>,
                              std::tuple<_BegT...>>::type;
};

//--------------------------------------------------------------------------------------//
//  Specialization for first sort operation
//
template <template <typename> class _Prio, typename _In, typename _Tp, typename... _InT>
struct SortT<_Prio, std::tuple<_In, _InT...>, std::tuple<>, std::tuple<_Tp>> {
  static constexpr bool value = (_Prio<_In>::value < _Prio<_Tp>::value);

  using type =
      typename std::conditional<(value),
                                typename SortT<_Prio, std::tuple<_InT...>, std::tuple<>,
                                               std::tuple<_In, _Tp>>::type,
                                typename SortT<_Prio, std::tuple<_InT...>, std::tuple<>,
                                               std::tuple<_Tp, _In>>::type>::type;
};

//--------------------------------------------------------------------------------------//
//  Specialization for second sort operation
//
template <template <typename> class _Prio, typename _In, typename _Ta, typename _Tb,
          typename... _BegT, typename... _InT>
struct SortT<_Prio, std::tuple<_In, _InT...>, std::tuple<_BegT...>,
             std::tuple<_Ta, _Tb>> {
  static constexpr bool iavalue = (_Prio<_In>::value < _Prio<_Ta>::value);
  static constexpr bool ibvalue = (_Prio<_In>::value < _Prio<_Tb>::value);
  static constexpr bool abvalue = (_Prio<_Ta>::value <= _Prio<_Tb>::value);

  using type = typename std::conditional<
      (iavalue),
      typename SortT<
          _Prio, std::tuple<_InT...>, Sort<_Prio, std::tuple<_BegT..., _In>>,
          conditional_t<(abvalue), std::tuple<_Ta, _Tb>, std::tuple<_Tb, _Ta>>>::type,
      typename SortT<_Prio, std::tuple<_InT...>, Sort<_Prio, std::tuple<_BegT..., _Ta>>,
                     conditional_t<(ibvalue), std::tuple<_In, _Tb>,
                                   std::tuple<_Tb, _In>>>::type>::type;
};

//--------------------------------------------------------------------------------------//
//  Specialization for all other sort operations after first and second
//
template <template <typename> class _Prio, typename _In, typename _Tp, typename... _InT,
          typename... _BegT, typename... _EndT>
struct SortT<_Prio, std::tuple<_In, _InT...>, std::tuple<_BegT...>,
             std::tuple<_Tp, _EndT...>> {
  static constexpr bool value = (_Prio<_In>::value < _Prio<_Tp>::value);

  using type = typename std::conditional<
      (value),
      typename SortT<_Prio, std::tuple<_InT...>, std::tuple<>,
                     std::tuple<_BegT..., _In, _Tp, _EndT...>>::type,
      typename SortT<_Prio, std::tuple<_In, _InT...>,
                     Sort<_Prio, Sort<_Prio, std::tuple<_BegT..., _Tp>>>,
                     std::tuple<_EndT...>>::type>::type;
};

//===--------------------------------------------------------------------------------===//
//              A list of all the particles for unrolling
//===--------------------------------------------------------------------------------===//

using ParticleTypes = std::tuple<CpuGamma, CpuElectron, GpuGamma, GpuElectron>;

//===--------------------------------------------------------------------------------===//
//              Priority Type-Traits
//===--------------------------------------------------------------------------------===//

template <typename _Tp>
struct PhysicsProcessAtRestPriority : std::integral_constant<int, 0> {};

template <typename _Tp>
struct PhysicsProcessAlongStepPriority : std::integral_constant<int, 0> {};

template <typename _Tp>
struct PhysicsProcessPostStepPriority : std::integral_constant<int, 0> {};

//===--------------------------------------------------------------------------------===//

template <>
struct PhysicsProcessPostStepPriority<ProxyStepLimiter>
    : std::integral_constant<int, -100> {};

template <>
struct PhysicsProcessPostStepPriority<ProxyScattering> : std::integral_constant<int, 0> {
};

template <>
struct PhysicsProcessPostStepPriority<ProxyTrackLimiter>
    : std::integral_constant<int, 10> {};

template <>
struct PhysicsProcessPostStepPriority<Transportation> : std::integral_constant<int, 100> {
};

//===--------------------------------------------------------------------------------===//
//              Type information class for Physics available to Particle
//===--------------------------------------------------------------------------------===//

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessList {
  using particle = ParticleType;
  using physics  = std::tuple<ProcessTypes...>;
};

//===--------------------------------------------------------------------------------===//
//              Generic stage expansion
//===--------------------------------------------------------------------------------===//

template <typename ParticleType, template <typename, typename> class StageType,
          typename... ProcessTypes>
struct PhysicsProcessStage {
  using type = std::tuple<StageType<ProcessTypes, ParticleType>...>;
};

template <typename ParticleType, template <typename, typename> class StageType,
          typename... ProcessTypes>
struct PhysicsProcessStage<ParticleType, StageType, std::tuple<ProcessTypes...>> {
  using type = std::tuple<StageType<ProcessTypes, ParticleType>...>;
};

//===--------------------------------------------------------------------------------===//
//              AtRest, AlongStep, PostStep stages
//===--------------------------------------------------------------------------------===//

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessAtRest
    : PhysicsProcessStage<
          ParticleType, AtRest,
          Sort<PhysicsProcessAtRestPriority, std::tuple<ProcessTypes...>>> {
  using ParticleTypeProcesses =
      Sort<PhysicsProcessAtRestPriority, std::tuple<ProcessTypes...>>;
};

//===--------------------------------------------------------------------------------===//

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessAlongStep
    : PhysicsProcessStage<
          ParticleType, AlongStep,
          Sort<PhysicsProcessAlongStepPriority, std::tuple<ProcessTypes...>>> {
  using process_type = Sort<PhysicsProcessAlongStepPriority, std::tuple<ProcessTypes...>>;
  using base_type    = PhysicsProcessStage<ParticleType, AlongStep, process_type>;
  using this_type    = PhysicsProcessAlongStep<ParticleType, ProcessTypes...>;

  template <typename _Track, typename AtRestApply_t, typename _Process>
  struct AtRestDoIt {
    template <typename _Proc = _Process, std::enable_if_t<(_Proc::EnableDoIt), int> = 0>
    AtRestDoIt(_Track *track)
    {
      GEANT_THIS_TYPE_TESTING_MARKER("");
      geantx::Log(kInfo) << GEANT_HERE << "[ALONG-STEP AT-REST DO-IT] "
                         << tim::demangle<_Proc>();
      _Process p(track);
      if (IsStopped(*track)) {
        if (IsAlive(*track)) Apply<void>::apply<AtRestApply_t>(track);
        // instead 'return' we want to say 'abort/break-out-of loop'
        // for now we 'repeat pointlessly the test in AlongStep struct (in
        // ProcessConcepts.hpp) return;
      }
    }

    template <typename _Proc = _Process, std::enable_if_t<!(_Proc::EnableDoIt), int> = 0>
    AtRestDoIt(_Track *track)
    {}
  };

  template <typename _Track, typename AtRestApply_t, typename _Process>
  struct ApplyAtRest_T;

  template <typename _Track, typename AtRestApply_t, template <typename...> class _Tuple,
            typename... _Process>
  struct ApplyAtRest_T<_Track, AtRestApply_t, _Tuple<_Process...>> {
    using type = _Tuple<AtRestDoIt<_Track, AtRestApply_t, _Process>...>;
  };

  template <typename _Track, typename AtRestApply_t>
  using ApplyAtRest =
      typename ApplyAtRest_T<_Track, AtRestApply_t, typename base_type::type>::type;
};

//===--------------------------------------------------------------------------------===//

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessPostStep
    : PhysicsProcessStage<
          ParticleType, PostStep,
          Sort<PhysicsProcessPostStepPriority, std::tuple<ProcessTypes...>>> {
  using process_type = Sort<PhysicsProcessPostStepPriority, std::tuple<ProcessTypes...>>;
  using base_type    = PhysicsProcessStage<ParticleType, PostStep, process_type>;
  using this_type    = PhysicsProcessPostStep<ParticleType, ProcessTypes...>;

  template <typename _Track, typename AtRestApply_t, typename _Process>
  struct AtRestDoIt {
    template <typename _Proc = _Process, std::enable_if_t<(_Proc::EnableDoIt), int> = 0>
    AtRestDoIt(_Track *track)
    {
      GEANT_THIS_TYPE_TESTING_MARKER("");
      geantx::Log(kInfo) << GEANT_HERE << "[POST-STEP AT-REST DO-IT] "
                         << tim::demangle<_Proc>();
      _Process p(track);
      if (IsStopped(*track)) {
        if (IsAlive(*track)) Apply<void>::apply<AtRestApply_t>(track);
        // instead 'return' we want to say 'abort/break-out-of loop'
        // for now we 'repeat pointlessly the test in AlongStep struct (in
        // ProcessConcepts.hpp) return;
      }
    }

    template <typename _Proc = _Process, std::enable_if_t<!(_Proc::EnableDoIt), int> = 0>
    AtRestDoIt(_Track *track)
    {}
  };

  template <typename _Track, typename AtRestApply_t, typename _Process>
  struct ApplyAtRest_T;

  template <typename _Track, typename AtRestApply_t, template <typename...> class _Tuple,
            typename... _Process>
  struct ApplyAtRest_T<_Track, AtRestApply_t, _Tuple<_Process...>> {
    using type = _Tuple<AtRestDoIt<_Track, AtRestApply_t, _Process>...>;
  };

  template <typename _Track, typename AtRestApply_t>
  using ApplyAtRest =
      typename ApplyAtRest_T<_Track, AtRestApply_t, typename base_type::type>::type;
};

//===--------------------------------------------------------------------------------===//
//              Specify the Physics for the Particles
//===--------------------------------------------------------------------------------===//

using CpuGammaPhysics =
    PhysicsProcessList<CpuGamma, ProxyScattering, ProxyStepLimiter, ProxyTrackLimiter,
                       ProxySecondaryGenerator, Transportation>;

using CpuElectronPhysics =
    PhysicsProcessList<CpuElectron, ProxyScattering, ProxyStepLimiter, ProxyTrackLimiter,
                       ProxySecondaryGenerator, Transportation>;

using GpuGammaPhysics =
    PhysicsProcessList<GpuGamma, ProxyScattering, ProxyStepLimiter, ProxyTrackLimiter,
                       ProxySecondaryGenerator, Transportation>;

using GpuElectronPhysics =
    PhysicsProcessList<GpuElectron, ProxyScattering, ProxyStepLimiter, ProxyTrackLimiter,
                       ProxySecondaryGenerator, Transportation>;

//===--------------------------------------------------------------------------------===//
//              A list of all particle + physics pairs
//===--------------------------------------------------------------------------------===//

using ParticlePhysicsTypes =
    std::tuple<CpuGammaPhysics, CpuElectronPhysics, GpuGammaPhysics, GpuElectronPhysics>;

//--------------------------------------------------------------------------------------//
