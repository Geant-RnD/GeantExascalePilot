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
 * @brief Assertion routines.
 */
//===----------------------------------------------------------------------===//

#pragma once

// Contract validation on input
#define REQUIRE(code)

// Internal consistency checking
#define CHECK(code)

// Contract validation on output
#define ENSURE(code)

// Always-on assertion that prints message if it fails
#define INSIST(code, msg_stream)
