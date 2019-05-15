/*!
 * \file   Geant/core/src/LoggerStatement.cpp
 * \note   Copyright (c) 2019 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */

#include "Geant/core/LoggerStatement.hpp"

#include "Geant/core/Assert.hpp"

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

  try {
    // Add a trailing newline
    *fMessage << '\n';

    // Get the string output
    const auto &message = fMessage->str();

    // Write it to all the streams
    for (auto *stream_ptr : fSinks) {
      // TODO: add a static mutex here to guarantee messages won't overlap.
      // Since we print the messages with a single call to operator<<,
      // hopefully they will generally be OK anyway.
      *stream_ptr << message << std::flush;
    }
  } catch (const std::exception &e) {
    std::cerr << "An error occurred writing a log message: " << e.what() << std::endl;
  }
}

//---------------------------------------------------------------------------//
} // end namespace geantx
