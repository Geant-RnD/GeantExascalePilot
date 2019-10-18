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
    PhysicsProcessList<CpuGamma, ProxyScattering, ProxySecondaryGenerator,
                       ProxyStepLimiter, ProxyTrackLimiter>;

using CpuElectronPhysics =
    PhysicsProcessList<CpuElectron, ProxyScattering, ProxySecondaryGenerator,
                       ProxyStepLimiter, ProxyTrackLimiter>;

using GpuGammaPhysics =
    PhysicsProcessList<GpuGamma, ProxyScattering, ProxySecondaryGenerator,
                       ProxyStepLimiter, ProxyTrackLimiter>;

using GpuElectronPhysics =
    PhysicsProcessList<GpuElectron, ProxyScattering, ProxySecondaryGenerator,
                       ProxyStepLimiter, ProxyTrackLimiter>;

//===----------------------------------------------------------------------===//
//              A list of all particle + physics pairs
//===----------------------------------------------------------------------===//

using ParticlePhysicsTypes =
    std::tuple<CpuGammaPhysics, CpuElectronPhysics, GpuGammaPhysics, GpuElectronPhysics>;
