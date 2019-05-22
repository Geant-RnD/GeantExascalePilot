/*!
 * \file   Geant/core/src/LoggerStatement.cpp
 * \note   Copyright (c) 2019 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */

#include "Geant/core/Config.hpp"
#include "Geant/core/LoggerStatement.hpp"

#include "PTL/ThreadPool.hh"

#include <mutex>

namespace geantx {
//---------------------------------------------------------------------------//
LoggerStatement::LoggerStatement(VecOstream streams) : fSinks(std::move(streams))
{
#ifdef REQUIRE_ON
  for (auto sink : fSinks) {
    REQUIRE(sink);
  }
#endif

  if (!fSinks.empty()) {
    // Allocate a message stream if we're actually doing output
    fMessage = std::make_unique<osstream_t>();
  }
  ENSURE(!fSinks.empty() == static_cast<bool>(fMessage));
}

//---------------------------------------------------------------------------//
LoggerStatement::~LoggerStatement() noexcept
{
  if (!fMessage) return;

#ifndef GEANT_CUDA_DEVICE_COMPILATION
  static std::mutex prntMutex;
#endif

  try {
    // Add a trailing newline
    *fMessage << '\n';

    // Get the string output
    const auto &message = fMessage->str();

    // Write it to all the streams
    for (auto *stream_ptr : fSinks) {
#ifndef GEANT_CUDA_DEVICE_COMPILATION
      std::lock_guard<std::mutex> lock(prntMutex);
#endif
      *stream_ptr << message << std::flush;
    }
  } catch (const std::exception &e) {
    std::cerr << "An error occurred writing a log message: " << e.what() << std::endl;
  }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return the current thread id
 */
uintmax_t GetThisThreadID()
{
  return PTL::ThreadPool::GetThisThreadID();
}

//---------------------------------------------------------------------------//
} // end namespace geantx
