//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file Geant/processes/Process.hpp
 * @brief Memory pool for device and host allocations.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Tuple.hpp"
#include "Geant/particles/Types.hpp"
#include "Geant/track/TrackAccessor.hpp"
#include "Geant/track/TrackState.hpp"

#include <random>
#include <string>

namespace geantx {
//----------------------------------------------------------------------------//
// This trait is marked by the source code as applicable to certain type
// of particle
//
template <typename ProcessType, typename ParticleType>
struct ProcessAvailable : std::false_type {};

//----------------------------------------------------------------------------//
// This trait is marked by the user code as applicable to certain type
// of particle
//
template <typename ProcessType, typename ParticleType>
struct ProcessEnabled : std::false_type {};

//----------------------------------------------------------------------------//
// This is the base class for a process
//
class Process {
public:
  Process(const std::string &name, const double &factor);
  Process()                = default;
  ~Process()               = default;
  Process(const Process &) = default;
  Process(Process &&)      = default;

  Process &operator=(const Process &) = delete;
  Process &operator=(Process &&) = default;

public:
  double GetPILFactor() const { return fPILfactor; }
  void SetPILFactor(const double &val) { fPILfactor = val; }

  const std::string &GetName() { return fName; }
  void SetName(const std::string &val) { fName = val; }

protected:
  double fPILfactor             = 0.0;
  std::string fName             = "unknown";
  static constexpr auto fMaxPIL = std::numeric_limits<double>::max();
};

//===----------------------------------------------------------------------===//
// I doubt someone will see this but if you do, please replace with whatever
// is needed to get a random number generator from VecMath.
//
// And if this is Philippe looking, consider me warned about static thread_local
// being slow.
//
//#if defined(GEANT_TESTING)
inline double get_rand()
{
  static auto _get_generator = []() {
    std::mt19937_64 gen(tim::get_env<uintmax_t>("GEANT_TESTING_SEED", 15432483));
    return gen;
  };
  static thread_local auto _gen = _get_generator();
  return std::generate_canonical<double, 10>(_gen);
}
//#endif
} // namespace geantx
