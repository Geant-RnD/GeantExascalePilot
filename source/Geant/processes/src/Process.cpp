//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/processes/src/Process.cpp
 * @brief Memory pool for device and host allocations.
 */
//===----------------------------------------------------------------------===//

#include "Geant/processes/Process.hpp"

namespace geantx {

GEANT_HOST
Process::Process(char* name)
      : fName(name)
  {}

GEANT_HOST
Process::Process()
    : fThreadId(-1)
{}

GEANT_HOST_DEVICE
Process::Process(int tid)
    : fThreadId(tid)
{}

} // namespace geantx
