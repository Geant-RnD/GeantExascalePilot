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
 * @brief Forward declaration of central types.
 */
//===----------------------------------------------------------------------===//

#pragma once

#include "Geant/core/Config.hpp"

namespace geantx {
inline namespace GEANT_IMPL_NAMESPACE {
class NullLoggerStatement {
  using This = NullLoggerStatement;

public:
  // >>> TYPEDEFS

public:
  NullLoggerStatement() {}
  ~NullLoggerStatement() noexcept = default;

  // Allow moving but not copying
  NullLoggerStatement(NullLoggerStatement &&) = default;
  NullLoggerStatement &operator=(NullLoggerStatement &&) = default;
  NullLoggerStatement(const NullLoggerStatement &)       = delete;
  NullLoggerStatement &operator=(const NullLoggerStatement &) = delete;

  /*!
   * \brief Ignore everything passed to us.
   */
  template <class T>
  This &operator<<(const T &)
  {
    return *this;
  }
};

} // namespace GEANT_IMPL_NAMESPACE
} // namespace geantx
