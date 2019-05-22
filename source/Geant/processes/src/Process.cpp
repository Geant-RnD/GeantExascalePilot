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

#include "Geant/processes/Process.hpp"

namespace geantx
{
Process::Process(const std::string& name, const double& factor)
: fPILfactor(factor)
, fName(name)
{
}

}  // namespace geantx