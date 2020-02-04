/*!
 * \file   geantx/core/Logger.hpp
 * \note   Copyright (c) 2019 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */

#pragma once

#include <iomanip>
#include <iostream>
#include <memory>

#ifndef __NVCC__
#  include "Geant/core/LoggerStatement.hpp"
#else
#  include "Geant/core/NullLoggerStatement.hpp"
#endif

namespace geantx {
// To be moved to its own header/source file.
uintmax_t GetThisThreadID();

//! Delayed evaluation of output for GEANT_HERE
struct StreamHere {
  const char *fFunction;
  const char *fFile;
  int fLine;
};

inline std::ostream &operator<<(std::ostream &os, const StreamHere &s)
{
  // this is a bit of a hack to make things more readable during initial dev
  // (J. Madsen)
  std::string file = s.fFile;
  if (file.find("/source/") != std::string::npos)
    file = file.substr(file.find("/source/") + 8);
  if (file.find("/include/") != std::string::npos)
    file = file.substr(file.find("/include/") + 9);
  static uintmax_t maxw = file.length();
  maxw                  = std::max<uintmax_t>(maxw, file.length());

  // Format string below "[%lu]> %s @ %s:%i "
  std::stringstream ss;

  ss << '[' << geantx::GetThisThreadID() << "]> " << std::setw(16) << std::left
     << s.fFunction << " @ " << std::setw(maxw) << file << ':' << std::setw(4) << s.fLine
     << ": ";
  os << ss.str();
  return os;
}

//========================================================= =================//
/*!
 * \macro GEANT_HERE
 * \brief To be used to allocate a log with the current location, including
 *        function name, file name and line number
 *
 * \code
      geantx::Log(kInfo) << GEANT_HERE << "some message";
 * \endcode
 */
#if !defined(GEANT_HERE)
#  define GEANT_HERE \
    StreamHere { __FUNCTION__, __FILE__, __LINE__ }
#endif

//---------------------------------------------------------------------------//
/*!
 * \page logging Logging messages in GeantX
 *
 * PRINT messages are fine-grained diagnostics.
 *
 * DIAGNOSTIC messages are for useful information that may produce a good bit
 * of output.
 *
 * STATUS messages should be essentially the same at a given program point, no
 * matter what the execution. (For example, "Building cross sections" or
 * "Transporting step 1".)
 *
 * INFO messages typically contain information unique to the particular
 * problem: e.g. "Loaded 123 cross sections" or "Set default foo to 123".
 *
 * WARNING messages should be about unexpected or unusual behavior.
 *
 * ERROR messages are when something went wrong but we're trying to recover
 * from it or ignore it.
 *
 * CRITICAL messages are meant for "the big goodbye": explaining the last error
 * message this process will give. The prefix "!*!*!" is actually intercepted
 * by Omnibus and used as a trigger to abort the MPI run if one processor
 * aborts.
 *
 * Example messages:
 * \code

   LogMaster(kPrint) << "Nuclide " << n << " has " << r << " reactions.";
   Log(kDiagnostic) << "Finished transporting on thread " << thread << ".";
   LogMaster(kStatus) << "Building solver...";
   LogMaster(kInfo) << "Built solution vector with " << num << " unknowns.";
   LogMaster(kWarning) << "Nuclide 1001 was remapped to 1000";
   Log(kError) << "Geometry error (lost particle) in history " << n;
   LogMaster(kFatal) << "Caught exception " << e.what() << "; aborting.";

 * \endcode
 */

//===========================================================================//
//! Enumeration for logging level.
enum LogLevel {
  kPrint = 0,  //!< Debugging messages
  kDiagnostic, //!< Diagnostics about current program execution
  kStatus,     //!< Program execution status (what stage is beginning)
  kInfo,       //!< Important informational messages
  kWarning,    //!< Warnings about unusual events
  kError,      //!< Something went wrong, but execution continues
  kFatal,      //!< Something went terribly wrong; we're aborting now! Bye!
  kEndLogLevel
};

#ifndef __NVCC__
//===========================================================================//
/*!
 * \class Logger
 * \brief Global parallel logging for geantx.
 *
 * This singleton class is generally accessed via the "Log" free function.
 *
 * Currently the thread ID is saved whenever the logger is instantiated (first
 * called), so if the communicator is changed, the original "master" thread will
 * be the only one that logs during a global call.
 *
 * The class is designed to replace: \code

    if (thread == 0) {
        cout << ">>> Global message" << endl;
    }

    cout << ">>> Encountered " << n << " cowboys on thread " << thread << endl;
    \endcode

 * with \code

    geantx::LogMaster() << "Global message" ;
    geantx::Log() << "Encountered " << n << " cowboys on thread "
                         << thread ;
 * \endcode
 *
 * The streams can be redirected at will by using the Logger accessor methods.
 *
 * \note The logging object returned by Log() will not evalute the arguments if
 * no output will be displayed.
 */
//===========================================================================//

class Logger {
public:
  //@{
  //! Type aliases
  using ostream_t = LoggerStatement::ostream_t;
  using SpOstream = std::shared_ptr<ostream_t>;
  //@}

private:
  // >>> DATA

  //! Local and global minimum log levels
  LogLevel fLocalLevel;
  LogLevel fGlobalLevel;

public:
  // >>> CONFIGURATION

  // Set MINIMUM verbosity level for local log calls to be logged.
  void SetLocalLevel(LogLevel level);

  // Set MINIMUM verbosity level for global log calls to be logged.
  void SetGlobalLevel(LogLevel level);

  // Set output stream from a shared pointer
  void Set(const std::string &key, SpOstream stream_sp, LogLevel min_level);

  // Set output stream from a raw pointer (UNSAFE EXCEPT WITH GLOBAL OSTREAM!)
  void Set(const std::string &key, ostream_t *stream_ptr, LogLevel min_level);

  // Remove an output stream
  void Remove(const std::string &key);

  // >>> STREAMING

  // Return a stream appropriate to the level for "master" thread output
  LoggerStatement GlobalStream(LogLevel level);

  // Return a stream appropriate to the level for local-thread output
  LoggerStatement LocalStream(LogLevel level);

  // >>> STATIC METHODS

  static Logger &GetInstance();

private:
  Logger();
  Logger(const Logger &);
  Logger &operator=(const Logger &);

  // >>> STATIC DATA

  // Prefixes for debug/info/etc e.g. "***"
  static const char *kLogPrefix[kEndLogLevel];

private:
  //! Struct for output levels
  struct Sink {
    std::string name;     //!< Name of output sink
    LogLevel level;       //!< Output only if message >= this level
    SpOstream stream_ptr; //!< SP to keep pointer alive

    Sink(const std::string &n, LogLevel lev) : name(n), level(lev), stream_ptr()
    {
      /* * */
    }
  };

  using VecOstream = LoggerStatement::VecOstream;

private:
  // Instead of doing something complicated like a sorted vector on name,
  // just have one sink for screen output, one for "log file" output
  Sink fScreenOutput;
  Sink fFileOutput;

  // Find the sink given this name
  Sink &Find(const std::string &key);

  // Build output streams based on the given level
  VecOstream BuildStreams(LogLevel level) const;
};

//---------------------------------------------------------------------------//
//! Return an ostream for global (master thread only) messages
inline LoggerStatement LogMaster(LogLevel level = kInfo)
{
  return Logger::GetInstance().GlobalStream(level);
}

//---------------------------------------------------------------------------//
//! Return an ostream for local messages
inline LoggerStatement Log(LogLevel level = kInfo)
{
  return Logger::GetInstance().LocalStream(level);
}
//---------------------------------------------------------------------------//
#else // for __NVCC__ defined:
//---------------------------------------------------------------------------//

class Logger;

inline NullLoggerStatement LogMaster(LogLevel level = kInfo)
{
  (void)sizeof(level);
  return {};
}

inline NullLoggerStatement Log(LogLevel level = kInfo)
{
  (void)sizeof(level);
  return {};
}

#endif
//---------------------------------------------------------------------------//
} // end namespace geantx
