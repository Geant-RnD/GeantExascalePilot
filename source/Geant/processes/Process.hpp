//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file
 * @brief Memory pool for device and host allocations.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

namespace geantx
{
class Process
{
public:
    Process() {}
    Process(const std::string& name, const double& factor);
    ~Process()              = default;
    Process(const Process&) = default;
    Process(Process&&)      = default;

    Process& operator=(const Process&) = delete;
    Process& operator=(Process&&) = default;

public:
    double GetPILFactor() const { return fPILfactor; }
    void   SetPILFactor(const double& val) { fPILfactor = val; }

    const std::string& GetName() { return fName; }
    void               SetName(const std::string& val) { fName = val; }

protected:
    double      fPILfactor = 0.0;
    std::string fName      = "unknown";
};
}  // namespace geantx
