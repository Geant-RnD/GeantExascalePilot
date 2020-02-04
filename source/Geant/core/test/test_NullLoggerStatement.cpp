//===------------------ GeantX --------------------------------------------===//
//
// Geant Exascale Pilot
//
// For the licensing terms see LICENSE file.
// For the list of contributors see CREDITS file.
// Copyright (C) 2019, Geant Exascale Pilot team,  All rights reserved.
//===----------------------------------------------------------------------===//
/**
 * @file source/Geant/core/test/test_Logger.cpp
 * @brief Test of NullLoggerStatement
 */
//===----------------------------------------------------------------------===//

#include "../NullLoggerStatement.hpp"

#include "gtest/gtest.h"
#include <atomic>

using namespace geantx;
using std::cout;
using std::endl;

class LoggerTest : public ::testing::Test {};

std::atomic<int> called_expensive{0};

struct Expensive {};

std::ostream &operator<<(std::ostream &os, const Expensive &)
{
  ++called_expensive;
  return os;
}

TEST_F(LoggerTest, general)
{
  NullLoggerStatement() << "Expensive: " << Expensive{} << "tada";
  EXPECT_EQ(0, called_expensive.load());
}
