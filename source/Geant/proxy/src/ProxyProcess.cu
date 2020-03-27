//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file ProxyProcess.cu
 * @brief base class for ProxyProcesses
 */
//===----------------------------------------------------------------------===//

#include "Geant/proxy/ProxyProcess.cuh"

namespace geantx {

GEANT_HOST
ProxyProcess::ProxyProcess()
    : fPILfactor(0.0), fThreadId(-1)
{}

GEANT_HOST_DEVICE
ProxyProcess::ProxyProcess(int tid)
    : fPILfactor(0.0),fThreadId(tid)
{}

} // namespace geantx
