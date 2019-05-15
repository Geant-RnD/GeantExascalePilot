//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//

#include "Geant/core/Error.hpp"
#ifdef USE_ROOT
#include "TError.h"
#endif
#include <stdarg.h>

namespace geantx {
inline namespace cxx {
void ErrorHandlerImpl(EMsgLevel level, const char *location, const char *msgfmt, ...)
{
  va_list args;
  va_start(args, msgfmt);

#ifdef USE_ROOT
  // Currently we use the ROOT message handler on the host/gcc code.
  ::ErrorHandler((int)level, location, msgfmt, args);
#else
  // Trivial implementation
  if (level > EMsgLevel::kPrint || (location == nullptr || location[0] == '\0')) {
    fprintf(stdout, "Geant Message level %d at %s:", (int)level, location);
  }
  vfprintf(stdout, msgfmt, args);
  fprintf(stdout, "\n");
#endif

  va_end(args);
}
} // namespace cxx
} // namespace geantx
