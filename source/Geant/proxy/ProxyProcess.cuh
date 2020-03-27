//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/proxy/ProxyProcess.hpp
 * @brief the base class of physics processes
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/Tuple.hpp"
#include "Geant/particles/Types.hpp"
#include "Geant/track/TrackAccessor.hpp"
#include "Geant/track/TrackState.hpp"

#include <random>
#include <string>

namespace geantx
{
//----------------------------------------------------------------------------//
// This trait is marked by the source code as applicable to certain type
// of particle
//
template <typename ProcessType, typename ParticleType>
struct ProcessAvailable : std::false_type
{};

//----------------------------------------------------------------------------//
// This trait is marked by the user code as applicable to certain type
// of particle
//
template <typename ProcessType, typename ParticleType>
struct ProcessEnabled : std::false_type
{};

//----------------------------------------------------------------------------//
// This is the base class for a process
//
class ProxyProcess
{
public:
    GEANT_HOST
    ProxyProcess();

    GEANT_HOST_DEVICE
    ProxyProcess(int tid);

    GEANT_HOST_DEVICE
    ~ProxyProcess() {}

/*
    ProxyProcess(const ProxyProcess&) = default;

    ProxyProcess(ProxyProcess&&)      = default;

    ProxyProcess& operator=(const ProxyProcess&) = delete;

    ProxyProcess& operator=(ProxyProcess&&) = default;
*/

public:
    GEANT_HOST_DEVICE
    double GetPILFactor() const { return fPILfactor; }

    GEANT_HOST_DEVICE
    void   SetPILFactor(const double& val) { fPILfactor = val; }

    GEANT_HOST_DEVICE
    inline void Print() { printf("GPU-GPU\n"); }

protected:
    int                   fThreadId;
    double                fPILfactor;
};

//===----------------------------------------------------------------------===//
}  // namespace geantx
