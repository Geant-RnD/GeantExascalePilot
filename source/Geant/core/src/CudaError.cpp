//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "Geant/core/CudaRuntime.hpp"
#include "Geant/core/Logger.hpp"

namespace geantx {
inline namespace cudaruntime {

//---------------------------------------------------------------------------//
/*!
 * \brief Log cuda error and throw an exception.
 */
static std::string PrepareMsgAndLog(const char *err_string, const char *msg,
                                    const char *function, const char *file, int line)
{
  std::stringstream ss;
  ss << "cudaCheckError() failed at ";
  if (function) ss << function << "@'";
  ss << file << "':" << line << " : " << err_string;
  if (msg) ss << "\n\tWhile executing:\n" << msg;

  Log(kError) << ss.str();
  return ss.str();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Log cuda error and throw an exception.
 */
void CudaError(const char *err_string, const char *msg, const char *file, int line)
{
  throw std::runtime_error(PrepareMsgAndLog(err_string, msg, nullptr, file, line));
}

//---------------------------------------------------------------------------//
/*!
 * \brief Log cuda error and throw an exception.
 */
void CudaErrorFunc(const char *err_string, const char *msg, const char *function,
                   const char *file, int line)
{
  throw std::runtime_error(PrepareMsgAndLog(err_string, msg, function, file, line));
}

//---------------------------------------------------------------------------//
/*!
 * \brief Log cuda error and throw an exception.
 */
void CudaError(const char *err_string, const char *file, int line)
{
  throw std::runtime_error(PrepareMsgAndLog(err_string, nullptr, nullptr, file, line));
}

//---------------------------------------------------------------------------//
/*!
 * \brief Log cuda error and throw an exception.
 */
void CudaErrorFunc(const char *err_string, const char *function, const char *file,
                   int line)
{
  throw std::runtime_error(PrepareMsgAndLog(err_string, nullptr, function, file, line));
}

} // namespace cudaruntime
} // namespace geantx