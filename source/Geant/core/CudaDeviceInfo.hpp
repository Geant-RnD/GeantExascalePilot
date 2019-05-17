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

#include "PTL/AutoLock.hh"
#include "PTL/Utility.hh"

#include <atomic>
#include <string>
#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <deque>

//======================================================================================//

namespace geantx {
inline namespace cuda {

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

} // namespace cuda

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
  {
  }

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

template <typename _Func, typename... _Args>
void RunAlgorithm(_Func cpu_func, _Func cuda_func, _Args... args)
{
  using PTL::GetEnv;
  bool use_cpu = GetEnv<bool>("GEANT_USE_CPU", false);
  if (use_cpu) {
    try {
      cpu_func(std::forward<_Args>(args)...);
    } catch (const std::exception &e) {
      PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
      std::cerr << e.what() << '\n';
    }
    return;
  }

  std::deque<DeviceOption> options;
  options.push_back(DeviceOption(0, "cpu", "Run on CPU"));
  options.push_back(DeviceOption(1, "gpu", "Run on GPU with CUDA"));

  std::string default_key = "gpu";

  auto default_itr =
      std::find_if(options.begin(), options.end(),
                   [&](const DeviceOption &itr) { return (itr == default_key); });

  //------------------------------------------------------------------------//
  auto print_options = [&]() {
    static bool first = true;
    if (!first)
      return;
    else
      first = false;

    std::stringstream ss;
    DeviceOption::Header(ss);
    for (const auto &itr : options) {
      ss << itr;
      if (itr == *default_itr) ss << "\t(default)";
      ss << "\n";
    }
    DeviceOption::Footer(ss);

    PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
    std::cout << "\n" << ss.str() << std::endl;
  };
  //------------------------------------------------------------------------//
  auto print_selection = [&](DeviceOption &selected_opt) {
    static bool first = true;
    if (!first)
      return;
    else
      first = false;

    std::stringstream ss;
    DeviceOption::Spacer(ss, '-');
    ss << "Selected device: " << selected_opt << "\n";
    DeviceOption::Spacer(ss, '-');

    PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
    std::cout << ss.str() << std::endl;
  };
  //------------------------------------------------------------------------//

  // Run on CPU if nothing available
  if (options.size() <= 1) {
    cpu_func(std::forward<_Args>(args)...);
    return;
  }

  // print the GPU execution type options
  print_options();

  default_key = default_itr->key;
  auto key    = GetEnv("GEANT_DEVICE", default_key);

  auto selection = std::find_if(options.begin(), options.end(),
                                [&](const DeviceOption &itr) { return (itr == key); });

  if (selection == options.end())
    selection = std::find_if(options.begin(), options.end(),
                             [&](const DeviceOption &itr) { return itr == default_key; });

  print_selection(*selection);

  try {
    switch (selection->index) {
    case 0:
      cpu_func(std::forward<_Args>(args)...);
      break;
    case 1:
      cuda_func(std::forward<_Args>(args)...);
      break;
    default:
      cpu_func(std::forward<_Args>(args)...);
      break;
    }
  } catch (std::exception &e) {
    if (selection != options.end() && selection->index != 0) {
      {
        PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
        std::cerr << "[TID: " << GetThisThreadID() << "] " << e.what() << std::endl;
        std::cerr << "[TID: " << GetThisThreadID() << "] "
                  << "Falling back to CPU algorithm..." << std::endl;
      }
      try {
        cpu_func(std::forward<_Args>(args)...);
      } catch (std::exception &_e) {
        std::stringstream ss;
        ss << "\n\nError executing :: " << _e.what() << "\n\n";
        {
          PTL::AutoLock l(PTL::TypeMutex<decltype(std::cout)>());
          std::cerr << _e.what() << std::endl;
        }
        throw std::runtime_error(ss.str().c_str());
      }
    }
  }
}

//======================================================================================//

} // namespace geantx

//======================================================================================//
