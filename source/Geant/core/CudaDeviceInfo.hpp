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
 * @brief Common macros and declaration
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"
#include "Geant/core/Logger.hpp"

#include <atomic>
#include <string>
#include <sstream>
#include <ostream>
#include <iomanip>

//======================================================================================//

namespace geantx {
inline namespace cudaruntime {

void DeviceQuery();

int DeviceCount();

inline int &ThisThreadDevice()
{
  // this creates a globally accessible function for determining the device
  // the thread is assigned to
  //
  static std::atomic<int> _ntid(0);
  static thread_local int _instance =
      (DeviceCount() > 0) ? ((_ntid++) % DeviceCount()) : 0;
  return _instance;
}

// this functions sets "this_thread_device()" value to device number
int SetDevice(int device);

// the functions below use "this_thread_device()" function to get device number
int MultiProcessorCount();

int MaxThreadsPerBlock();

int WarpSize();

int SharedMemoryPerBlock();

//--------------------------------------------------------------------------------------//

} // namespace cudaruntime

//--------------------------------------------------------------------------------------//

struct DeviceOption {
  //
  //  This class enables the selection of a device at runtime
  //
public:
  typedef std::string string_t;
  typedef const string_t &crstring_t;

  int index;
  string_t key;
  string_t description;

  DeviceOption(const int &_idx, crstring_t _key, crstring_t _desc)
      : index(_idx), key(_key), description(_desc)
  {}

  static void Spacer(std::ostream &os, const char c = '-')
  {
    std::stringstream ss;
    ss.fill(c);
    ss << std::setw(90) << ""
       << "\n";
    os << ss.str();
  }

  friend bool operator==(const DeviceOption &lhs, const DeviceOption &rhs)
  {
    return (lhs.key == rhs.key && lhs.index == rhs.index);
  }

  friend bool operator==(const DeviceOption &itr, crstring_t cmp)
  {
    return (!IsNumeric(cmp)) ? (itr.key == ToLower(cmp))
                             : (itr.index == FromString<int>(cmp));
  }

  friend bool operator!=(const DeviceOption &lhs, const DeviceOption &rhs)
  {
    return !(lhs == rhs);
  }

  friend bool operator!=(const DeviceOption &itr, crstring_t cmp)
  {
    return !(itr == cmp);
  }

  static void Header(std::ostream &os)
  {
    std::stringstream ss;
    ss << "\n";
    Spacer(ss, '=');
    ss << "Available GPU options:\n";
    ss << "\t" << std::left << std::setw(5) << "INDEX"
       << "  \t" << std::left << std::setw(12) << "KEY"
       << "  " << std::left << std::setw(40) << "DESCRIPTION"
       << "\n";
    os << ss.str();
  }

  static void Footer(std::ostream &os)
  {
    std::stringstream ss;
    ss << "\nTo select an option for runtime, set GEANT_DEVICE_TYPE "
       << "environment variable\n  to an INDEX or KEY above\n";
    Spacer(ss, '=');
    os << ss.str();
  }

  friend std::ostream &operator<<(std::ostream &os, const DeviceOption &opt)
  {
    std::stringstream ss;
    ss << "\t" << std::right << std::setw(5) << opt.index << "  \t" << std::left
       << std::setw(12) << opt.key << "  " << std::left << std::setw(40)
       << opt.description;
    os << ss.str();
    return os;
  }

  // helper function for converting to lower-case
  inline static std::string ToLower(std::string val)
  {
    for (auto &itr : val)
      itr = static_cast<char>(::tolower(itr));
    return val;
  }

  // helper function to convert string to another type
  template <typename _Tp>
  static _Tp FromString(const std::string &val)
  {
    std::stringstream ss;
    _Tp ret;
    ss << val;
    ss >> ret;
    return ret;
  }

  // helper function to determine if numeric represented as string
  inline static bool IsNumeric(const std::string &val)
  {
    if (val.length() > 0) {
      auto f = val.find_first_of("0123456789");
      if (f == std::string::npos) // no numbers
        return false;
      auto l = val.find_last_of("0123456789");
      if (val.length() <= 2) // 1, 2., etc.
        return true;
      else
        return (f != l); // 1.0, 1e3, 23, etc.
    }
    return false;
  }
};

//======================================================================================//

} // namespace geantx

//======================================================================================//
