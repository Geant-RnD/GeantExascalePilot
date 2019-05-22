/*!
 * \file   Geant/core/LoggerStatement.hpp
 * \note   Copyright (c) 2019 Oak Ridge National Laboratory, UT-Battelle, LLC.
 */

#pragma once

#include "Geant/core/Config.hpp"

#include <vector>
#include <sstream>
#include <iostream>
#include <memory>

namespace geantx {

//===========================================================================//
/*!
 * \class LoggerStatement
 * \brief Support class for Logger that emulates an ostream.
 *
 * This class is designed to intercept ostream-like output, save it to a
 * buffer, and send it to multiple streams when it reaches the end of its
 * scope.
 *
 * It should never be stored; its lifetime should be the the scope of the
 * single statement from which it's created.
 *
 * The vector of string arguments are "reference"-like pointers: they should be
 * temporary (lifespan of the logger statement) and will not be deleted.
 */
//===========================================================================//

class LoggerStatement {
  using This = LoggerStatement;

public:
  // >>> TYPEDEFS

  //! Output string type
  using ostream_t = std::ostream;

  //! Vector of pointers to output streams
  using VecOstream = std::vector<ostream_t *>;

  //! Function signature for a stream maniupulator (such as endl)
  using StreamManipulator = ostream_t &(*)(ostream_t &);

private:
  // >>> DATA

  //! String stream type compatible with ostream type
  using osstream_t =
      std::basic_ostringstream<ostream_t::char_type, ostream_t::traits_type>;

  //! String stream for saving this log message
  std::unique_ptr<osstream_t> fMessage;

  //! Vector of "sinks" to output to
  VecOstream fSinks;

public:
  // Construct with streams to send message to
  explicit LoggerStatement(VecOstream streams);

  // Send message on destruction
  ~LoggerStatement() noexcept;

  // Allow moving but not copying
  LoggerStatement(LoggerStatement &&) = default;
  LoggerStatement &operator=(LoggerStatement &&) = default;
  LoggerStatement(const LoggerStatement &)       = delete;
  LoggerStatement &operator=(const LoggerStatement &) = delete;

  /*!
   * \brief Add a prefix to the statement.
   *
   */
  This &From(const char *msg)
  {
    *this << msg << " :";
    return *this;
  }

  /*!
   * \brief Act like an ostream, but return ourself.
   *
   * This allows us to intelligently disable writing expensive operations to
   * the stream if they're not going to be output. If we're saving output,
   * write the given data to the string stream.
   */
  template <class T>
  This &operator<<(const T &rhs)
  {
    if (fMessage) {
      *fMessage << rhs;
    }
    return *this;
  }

  /*!
   * \brief Specialization on const char* to reduce object size.
   */
  This &operator<<(const char *rhs)
  {
    if (fMessage) {
      *fMessage << rhs;
    }
    return *this;
  }

  /*!
   * \brief Accept manipulators such as std::endl.
   *
   * This allows us to intelligently disable writing expensive operations to
   * the stream if they're not going to be output.
   */
  This &operator<<(StreamManipulator manip)
  {
    if (fMessage) {
      manip(*fMessage);
    }
    return *this;
  }
};

} // namespace geantx
