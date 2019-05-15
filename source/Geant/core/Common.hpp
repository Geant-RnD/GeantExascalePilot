// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "Geant/core/Macros.hpp"

#include <atomic>
#include <string>
#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>

//======================================================================================//

namespace geantx {
//--------------------------------------------------------------------------------------//

inline namespace cuda {
void device_query();
int device_count();

inline int &this_thread_device()
{
  // this creates a globally accessible function for determining the device
  // the thread is assigned to
  //
  static std::atomic<int> _ntid(0);
  static thread_local int _instance =
      (device_count() > 0) ? ((_ntid++) % device_count()) : 0;
  return _instance;
}

// this functions sets "this_thread_device()" value to device number
int set_device(int device);

// the functions below use "this_thread_device()" function to get device number
int multi_processor_count();

int max_threads_per_block();

int warp_size();

int shared_memory_per_block();

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

  static void spacer(std::ostream &os, const char c = '-')
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
    return (!is_numeric(cmp)) ? (itr.key == tolower(cmp))
                              : (itr.index == from_string<int>(cmp));
  }

  friend bool operator!=(const DeviceOption &lhs, const DeviceOption &rhs)
  {
    return !(lhs == rhs);
  }

  friend bool operator!=(const DeviceOption &itr, crstring_t cmp)
  {
    return !(itr == cmp);
  }

  static void header(std::ostream &os)
  {
    std::stringstream ss;
    ss << "\n";
    spacer(ss, '=');
    ss << "Available GPU options:\n";
    ss << "\t" << std::left << std::setw(5) << "INDEX"
       << "  \t" << std::left << std::setw(12) << "KEY"
       << "  " << std::left << std::setw(40) << "DESCRIPTION"
       << "\n";
    os << ss.str();
  }

  static void footer(std::ostream &os)
  {
    std::stringstream ss;
    ss << "\nTo select an option for runtime, set GEANT_DEVICE_TYPE "
       << "environment variable\n  to an INDEX or KEY above\n";
    spacer(ss, '=');
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
  inline static std::string tolower(std::string val)
  {
    for (auto &itr : val)
      itr = scast<char>(::tolower(itr));
    return val;
  }

  // helper function to convert string to another type
  template <typename _Tp>
  static _Tp from_string(const std::string &val)
  {
    std::stringstream ss;
    _Tp ret;
    ss << val;
    ss >> ret;
    return ret;
  }

  // helper function to determine if numeric represented as string
  inline static bool is_numeric(const std::string &val)
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
void run_algorithm(_Func cpu_func, _Func cuda_func, _Args... args)
{
  bool use_cpu = GetEnv<bool>("GEANT_USE_CPU", false);
  if (use_cpu) {
    try {
      cpu_func(_Forward_t(_Args, args));
    } catch (const std::exception &e) {
      AutoLock l(TypeMutex<decltype(std::cout)>());
      std::cerr << e.what() << '\n';
    }
    return;
  }

  std::deque<DeviceOption> options;
  options.push_back(DeviceOption(0, "cpu", "Run on CPU"));

#if defined(GEANT_USE_GPU)
#if defined(GEANT_USE_CUDA)
  options.push_back(DeviceOption(1, "gpu", "Run on GPU with CUDA"));
  options.push_back(DeviceOption(2, "cuda", "Run on GPU with CUDA (deprecated)"));
#endif
#endif

#if defined(GEANT_USE_GPU) && defined(GEANT_USE_CUDA)
  std::string default_key = "gpu";
#else
  std::string default_key = "cpu";
#endif

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
    DeviceOption::header(ss);
    for (const auto &itr : options) {
      ss << itr;
      if (itr == *default_itr) ss << "\t(default)";
      ss << "\n";
    }
    DeviceOption::footer(ss);

    AutoLock l(TypeMutex<decltype(std::cout)>());
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
    DeviceOption::spacer(ss, '-');
    ss << "Selected device: " << selected_opt << "\n";
    DeviceOption::spacer(ss, '-');

    AutoLock l(TypeMutex<decltype(std::cout)>());
    std::cout << ss.str() << std::endl;
  };
  //------------------------------------------------------------------------//

  // Run on CPU if nothing available
  if (options.size() <= 1) {
    cpu_func(_Forward_t(_Args, args));
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
      cpu_func(_Forward_t(_Args, args));
      break;
    case 1:
      cuda_func(_Forward_t(_Args, args));
      break;
    default:
      cpu_func(_Forward_t(_Args, args));
      break;
    }
  } catch (std::exception &e) {
    if (selection != options.end() && selection->index != 0) {
      {
        AutoLock l(TypeMutex<decltype(std::cout)>());
        std::cerr << "[TID: " << GetThisThreadID() << "] " << e.what() << std::endl;
        std::cerr << "[TID: " << GetThisThreadID() << "] "
                  << "Falling back to CPU algorithm..." << std::endl;
      }
      try {
        cpu_func(_Forward_t(_Args, args));
      } catch (std::exception &_e) {
        std::stringstream ss;
        ss << "\n\nError executing :: " << _e.what() << "\n\n";
        {
          AutoLock l(TypeMutex<decltype(std::cout)>());
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
