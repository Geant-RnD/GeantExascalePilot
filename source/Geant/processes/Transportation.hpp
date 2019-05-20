//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Transportation.hpp
 * @brief Transportation process
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Tuple.hpp"
#include "Geant/processes/Process.hpp"

#include <type_traits>

namespace geantx
{
// TODO: avoid having to specify all particle types. Instead we will create
// an "AnyParticle" type or find a way for all the possible particle types to
// be expanded and/or added to
//
class Electron;
class Gamma;
class KaonLong;
class KaonMinus;
class KaonPlus;
class KaonShort;
class KaonZero;
class Neutron;
class PionMinus;
class PionPlus;
class PionZero;
class Positron;
class Proton;

class Transportation : public Process
{
public:
    // Enable/disable GetPhysicalInteractionLength (GPIL) functions
    static constexpr bool EnableAtRestGPIL    = false;
    static constexpr bool EnableAlongStepGPIL = true;
    static constexpr bool EnablePostStepGPIL  = false;
    // Enable/disable DoIt functions
    static constexpr bool EnableAtRestDoIt    = false;
    static constexpr bool EnableAlongStepDoIt = true;
    static constexpr bool EnablePostStepDoIt  = true;

    // see TODO above
    using AllowedParticles =
        Tuple<Electron, Gamma, KaonLong, KaonMinus, KaonPlus, KaonShort, KaonZero,
              Neutron, PionMinus, PionPlus, PionZero, Positron, Proton>;

    // User provides:

public:
    Transportation();
    ~Transportation();

    // here the transportation proposed a step distance
    double AlongStepGPIL();
    // double PostStepGPIL();

    void AlongStepDoIt();
    void PostStepDoIt();
};

template <typename ParticleType>
struct ProcessAvailable<Transportation, ParticleType> : std::true_type
{
};

template <typename ParticleType>
struct ProcessEnabled<Transportation, ParticleType> : std::true_type
{
};

}  // namespace geantx
