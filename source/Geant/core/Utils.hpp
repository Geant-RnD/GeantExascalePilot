//===------------------ Geant-V ---------------------------------*- C++ -*-===//
//
//                     Geant-V Prototype
//
//===----------------------------------------------------------------------===//
/**
 * @file GeantError.h
 * @brief Error handling routines.
 * @author Philippe Canal
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include <libgen.h>
#include <string>

namespace geant {
inline namespace GEANT_IMPL_NAMESPACE {
inline std::string GetDataFileLocation(int argc, char **argv, const char *dataFilename)
{
  const char *dir = argc > 0 ? dirname(argv[0]) : ".";

  if (dir == nullptr) dir = ".";
  return std::string(dir) + "/" + dataFilename;
}

inline std::string GetDataFileLocation(int argc, char **argv, std::string dataFilename)
{
  return GetDataFileLocation(argc, argv, dataFilename.c_str());
}

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geant
