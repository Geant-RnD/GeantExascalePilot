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

#include "Geant/processes/ProcessConcepts.hpp"
#include "Geant/processes/Transportation.hpp"

#include "Geant/proxy/ProxyParticles.hpp"
#include "Geant/proxy/ProxyScattering.hpp"
#include "Geant/proxy/ProxySecondaryGenerator.hpp"
#include "Geant/proxy/ProxyStepLimiter.hpp"
#include "Geant/proxy/ProxyTrackLimiter.hpp"

using namespace geantx;

//===----------------------------------------------------------------------===//
//              A list of all the particles for unrolling
//===----------------------------------------------------------------------===//

using ParticleTypes = std::tuple<CpuGamma, CpuElectron, GpuGamma, GpuElectron>;

//===----------------------------------------------------------------------===//
//              Type information class for Physics available to Particle
//===----------------------------------------------------------------------===//

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessList
{
    using particle = ParticleType;
    using physics  = std::tuple<ProcessTypes...>;
};

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessAtRest
{};

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessAlongStep
{};

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessPostStep
{};

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessAtRest<ParticleType, std::tuple<ProcessTypes...>>
{
    using type = std::tuple<AtRest<ProcessTypes, ParticleType>...>;
};

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessAlongStep<ParticleType, std::tuple<ProcessTypes...>>
{
    using type = std::tuple<AlongStep<ProcessTypes, ParticleType>...>;
};

template <typename ParticleType, typename... ProcessTypes>
struct PhysicsProcessPostStep<ParticleType, std::tuple<ProcessTypes...>>
{
    using type = std::tuple<PostStep<ProcessTypes, ParticleType>...>;
};

//===----------------------------------------------------------------------===//
//              Specify the Physics for the Particles
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
//              A list of all particle + physics pairs
//===----------------------------------------------------------------------===//

using ParticlePhysicsTypes =
    std::tuple<CpuGammaPhysics, CpuElectronPhysics, GpuGammaPhysics, GpuElectronPhysics>;

//===----------------------------------------------------------------------===//
//              Priority Type-Traits
//===----------------------------------------------------------------------===//

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

//
//      In general, this is still a WIP -- writing a quicksort of templates
//      is not trivial...
//

template <typename _Tp>
struct PhysicsProcessAtRestPriority : std::integral_constant<int, 0>
{};

template <typename _Tp>
struct PhysicsProcessAlongStepPriority : std::integral_constant<int, 0>
{};

template <typename _Tp>
struct PhysicsProcessPostStepPriority : std::integral_constant<int, 0>
{};

template <>
struct PhysicsProcessPostStepPriority<Transportation> : std::integral_constant<int, -100>
{};

template <>
struct PhysicsProcessPostStepPriority<ProxyStepLimiter> : std::integral_constant<int, 100>
{};

template <>
struct PhysicsProcessPostStepPriority<ProxyScattering> : std::integral_constant<int, 10>
{};

//===----------------------------------------------------------------------===//
//              Sorting the Type-Traits
//===----------------------------------------------------------------------===//

template <typename _Tp>
struct PhysicsProcessPriority
{
    static constexpr intmax_t AtRestPriority = PhysicsProcessAtRestPriority<_Tp>::value;
    static constexpr intmax_t AlongStepPriority =
        PhysicsProcessAlongStepPriority<_Tp>::value;
    static constexpr intmax_t PostStepPriority =
        PhysicsProcessPostStepPriority<_Tp>::value;
};

template <typename _Lhs, typename _Rhs>
struct SortPhysicsProcessAtRest
: std::conditional<(PhysicsProcessPriority<_Lhs>::AtRestPriority >
                    PhysicsProcessPriority<_Rhs>::AtRestPriority),
                   std::true_type, std::false_type>::type
{};

template <typename _Lhs, typename _Rhs>
struct SortPhysicsProcessAlongStep
: std::conditional<(PhysicsProcessPriority<_Lhs>::AlongStepPriority >
                    PhysicsProcessPriority<_Rhs>::AlongStepPriority),
                   std::true_type, std::false_type>::type
{};

template <typename _Lhs, typename _Rhs>
struct SortPhysicsProcessPostStep
: std::conditional<(PhysicsProcessPriority<_Lhs>::PostStepPriority >
                    PhysicsProcessPriority<_Rhs>::PostStepPriority),
                   std::true_type, std::false_type>::type
{};

/*
struct sort
{
    struct meta
    {
        template <typename T, T, typename>
        struct prepend;

        template <typename T, T Add, template <T...> class Z, T... Is>
        struct prepend<T, Add, Z<Is...>>
        {
            using type = Z<Add, Is...>;
        };

        template <typename T, typename Pack1, typename Pack2>
        struct concat;

        template <typename T, template <T...> class Z, T... Ts, T... Us>
        struct concat<T, Z<Ts...>, Z<Us...>>
        {
            using type = Z<Ts..., Us...>;
        };
    };

    template <typename _Tp, typename _Up>
    struct less_than
    : std::conditional<(_Tp::value < _Up::value), std::true_type, std::false_type>::type
    {};

    template <template <typename> class T, typename Types>
    struct sort;

    template <typename T, template <T...> class Z>
    struct sort<T, Z<>, Comparator>
    {
        using type = Z<>;
    };

    template <typename T, typename Pack, template <T> class UnaryPredicate>
    struct filter;

    template <typename T, template <T...> class Z, template <T> class UnaryPredicate, T I,
              T... Is>
    struct filter<T, Z<I, Is...>, UnaryPredicate>
    {
        using type = typename std::conditional<
            UnaryPredicate<I>::value,
            typename meta::prepend<
                T, I, typename filter<T, Z<Is...>, UnaryPredicate>::type>::type,
            typename filter<T, Z<Is...>, UnaryPredicate>::type>::type;
    };

    template <typename T, template <T...> class Z, template <T> class UnaryPredicate>
    struct filter<T, Z<>, UnaryPredicate>
    {
        using type = Z<>;
    };

    template <typename T, template <T...> class Z, T N, T... Is,
              template <T, T> class Comparator>
    struct sort<T, Z<N, Is...>, Comparator>
    {
        // Using the quicksort method.
        template <T I>
        struct less_than : std::integral_constant<bool, Comparator<I, N>::value>
        {};
        template <T I>
        struct more_than : std::integral_constant<bool, !Comparator<I, N>::value>
        {};
        using subsequence_less_than_N = typename filter<T, Z<Is...>, less_than>::type;
        using subsequence_more_than_N = typename filter<T, Z<Is...>, more_than>::type;
        using type                    = typename meta::concat<
            T, typename sort<T, subsequence_less_than_N, Comparator>::type,
            typename meta::prepend<
                T, N,
                typename sort<T, subsequence_more_than_N, Comparator>::type>::type>::type;
    };
};
*/

template <typename... Types>
struct IsEmpty
: std::conditional<(sizeof...(Types) > 0), std::true_type, std::false_type>::type
{};

template <typename... Types>
struct IsEmpty<std::tuple<Types...>>
: std::conditional<(sizeof...(Types) > 0), std::true_type, std::false_type>::type
{};

template <typename List>
struct FrontT
{
    using type = List;
};

template <typename Head, typename... Tail>
struct FrontT<std::tuple<Head, Tail...>>
{
    using type = Head;
};

template <typename List>
using Front = typename FrontT<List>::type;

template <typename List, typename NewElement>
struct PushFrontT;

template <typename... Elements, typename NewElement>
struct PushFrontT<std::tuple<Elements...>, NewElement>
{
    using type = std::tuple<NewElement, Elements...>;
};

template <typename List, typename NewElement>
using PushFront = typename PushFrontT<List, NewElement>::type;

// yield T when using member Type:
template <typename T>
struct IdentityT
{
    using type = T;
};

template <typename List, typename Element,
          template <typename T, typename U> class Compare, bool = IsEmpty<List>::value>
struct InsertSortedT;

template <typename List, typename Element,
          template <typename T, typename U> class Compare>
struct InsertSortedT<List, Element, Compare, false>
{
    // compute the tail of the resulting list:
    using tail =
        typename std::conditional<Compare<Element, Front<List>>::value, IdentityT<List>,
                                  InsertSortedT<PopFront<List>, Element, Compare>>::type;
    // compute the head of the resulting list:
    using head =
        std::conditional<Compare<Element, Front<List>>::value, Element, Front<List>>;

    using type = PushFront<head, tail>;
};

template <typename List, typename Element,
          template <typename T, typename U> class Compare>
struct InsertSortedT<List, Element, Compare, true>
{
    using type = std::tuple<Element>;
};

template <typename List, typename Element,
          template <typename T, typename U> class Compare>
using InsertSorted = typename InsertSortedT<List, Element, Compare>::type;
