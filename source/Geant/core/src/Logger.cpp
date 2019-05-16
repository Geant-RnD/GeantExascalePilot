/*!
 * \file   geantx/core/Logger.cpp
 * \note   Copyright (c) 2019 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */

#include "Geant/core/Logger.hpp"

#include "Geant/core/Assert.hpp"

#include "PTL/ThreadPool.hh"

//---------------------------------------------------------------------------//
// ANONYMOUS HELPER FUNCTIONS
//---------------------------------------------------------------------------//
namespace {
//! Custom deleter so that a regular pointer can be wrapped in an SP
struct null_deleter {
  void operator()(const void *)
  { /* * */
  }
};
} // end anonymous namespace

namespace geantx {
//---------------------------------------------------------------------------//
// STATIC DATA
//---------------------------------------------------------------------------//
const char *Logger::kLogPrefix[kEndLogLevel] = {
    "",       //  kPrint
    "",       //  kDiagnostic
    "::: ",   //  kStatus
    ">>> ",   //  kInfo
    "*** ",   //  kWarning
    "!!! ",   //  kError
    "!*!*! ", //  kFatal
};

//---------------------------------------------------------------------------//
// LOGGER
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
Logger::Logger()
    : fLocalLevel(kDiagnostic), fGlobalLevel(kDiagnostic),
      fScreenOutput("screen", kPrint), fFileOutput("file", kEndLogLevel)
{
  // Default screen output is cerr
  fScreenOutput.stream_ptr.reset(&std::cerr, null_deleter());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set verbosity level for local log calls.
 */
void Logger::SetLocalLevel(LogLevel level)
{
  REQUIRE(level < kEndLogLevel);
  fLocalLevel = level;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set verbosity level for global log calls.
 */
void Logger::SetGlobalLevel(LogLevel level)
{
  REQUIRE(level < kEndLogLevel);
  fGlobalLevel = level;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set an output handle
 *
 * \warning This is UNSAFE except with global ostreams such as cout!
 *
 * Since the Logger is global, it will almost certainly exceed the scope of any
 * local stream, leading to dereferencing of deallocated data.
 *
 * If you absolutely must give a raw pointer here, make *sure* to call
 * \c remove() on it when the pointer's reference is destroyed.
 */
void Logger::Set(const std::string &key, ostream_t *stream_ptr, LogLevel min_level)
{
  REQUIRE(stream_ptr);
  REQUIRE(min_level < kEndLogLevel);

  Sink &sink = this->Find(key);

  sink.name       = key;
  sink.level      = min_level;
  sink.stream_ptr = SpOstream(stream_ptr, null_deleter());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set an output handle from a shared pointer.
 *
 * This is the preferred way to set a logger handle because it has
 * reference-counted semantics.
 *
 * Note that because of static initialization order, it's generally bad to
 * have logger output during destructors of other static objects -- this logger
 * instance could be deleted before those other objects.
 */
void Logger::Set(const std::string &key, SpOstream stream_sp, LogLevel min_level)
{
  REQUIRE(stream_sp);
  REQUIRE(min_level < kEndLogLevel);

  Sink &sink = this->Find(key);

  sink.name       = key;
  sink.level      = min_level;
  sink.stream_ptr = stream_sp;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Remove an output handle.
 */
void Logger::Remove(const std::string &key)
{
  Sink &sink = this->Find(key);
  sink.stream_ptr.reset();
}

//---------------------------------------------------------------------------//
// ACCESSORS
//---------------------------------------------------------------------------//
/*!
 * \brief Return a stream appropriate to the level for node-zero output
 */
LoggerStatement Logger::GlobalStream(LogLevel level)
{
  REQUIRE(level < kEndLogLevel);

  LoggerStatement::VecOstream streams;

  // Only add streams on node zero
  if (level >= fGlobalLevel && PTL::ThreadPool::GetThisThreadID() == 0) {
    streams = this->BuildStreams(level);
  }

  // Create the logger statement (moving the vec streams for efficiency)
  LoggerStatement result(std::move(streams));

  // Pipe prefix to the stream before returning
  result << kLogPrefix[level];

  // Return the expiring LoggerStatement, implicit move
  return result;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return a stream appropriate to the level for local-node output
 */
LoggerStatement Logger::LocalStream(LogLevel level)
{
  REQUIRE(level < kEndLogLevel);

  LoggerStatement::VecOstream streams;

  if (level >= fLocalLevel) {
    streams = this->BuildStreams(level);
  }

  // Create the logger statement (moving the vec streams for efficiency)
  LoggerStatement result(std::move(streams));

  // Pipe prefix to the stream before returning
  result << kLogPrefix[level];

  // Return the expiring LoggerStatement, implicit move
  return result;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Remove an output handle.
 */
Logger::Sink &Logger::Find(const std::string &key)
{
  if (key == "screen") {
    return fScreenOutput;
  } else if (key == "file") {
    return fFileOutput;
  } else {
    INSIST(false, "Currently only screen and file are supported log keys; '"
                      << key << "' is invalid.");
  }

  // Squelch compiler errors
  return this->Find(key);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build output streams based on the given level
 */
Logger::VecOstream Logger::BuildStreams(LogLevel level) const
{
  VecOstream streams;

  for (const Sink *s : {&fScreenOutput, &fFileOutput}) {
    CHECK(s);
    if (s->stream_ptr && (level >= s->level)) {
      streams.push_back(s->stream_ptr.get());
    }
  }
  return streams;
}

//---------------------------------------------------------------------------//
// STATIC METHODS
//---------------------------------------------------------------------------//
/*!
 * \brief Access global logging instance.
 */
Logger &Logger::GetInstance()
{
  static Logger s_instance;
  CHECK(std::end(kLogPrefix) - std::begin(kLogPrefix) == kEndLogLevel);
  return s_instance;
}

} // end namespace geantx
