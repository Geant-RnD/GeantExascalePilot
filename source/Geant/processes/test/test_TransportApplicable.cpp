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

#include <Geant/particles/Electron.hpp>
#include <Geant/particles/Gamma.hpp>
#include <Geant/particles/KaonLong.hpp>
#include <Geant/particles/KaonMinus.hpp>
#include <Geant/particles/KaonPlus.hpp>
#include <Geant/particles/KaonShort.hpp>
#include <Geant/particles/KaonZero.hpp>
#include <Geant/particles/Neutron.hpp>
#include <Geant/particles/Particle.hpp>
#include <Geant/particles/PhysicsParameters.hpp>
#include <Geant/particles/PionMinus.hpp>
#include <Geant/particles/PionPlus.hpp>
#include <Geant/particles/PionZero.hpp>
#include <Geant/particles/Positron.hpp>
#include <Geant/particles/Proton.hpp>
#include <Geant/processes/Transportation.hpp>

#include "gtest/gtest.h"
#include <atomic>

using namespace geantx;
using std::cout;
using std::endl;

class TransportApplicable : public ::testing::Test {};

TEST_F(TransportApplicable, Electron)
{
  auto value = Transportation::IsApplicable<Electron>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, Gamma)
{
  auto value = Transportation::IsApplicable<Gamma>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, KaonLong)
{
  auto value = Transportation::IsApplicable<KaonLong>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, KaonMinus)
{
  auto value = Transportation::IsApplicable<KaonMinus>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, KaonPlus)
{
  auto value = Transportation::IsApplicable<KaonPlus>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, KaonShort)
{
  auto value = Transportation::IsApplicable<KaonShort>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, KaonZero)
{
  auto value = Transportation::IsApplicable<KaonZero>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, Neutron)
{
  auto value = Transportation::IsApplicable<Neutron>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, Particle)
{
  auto value = Transportation::IsApplicable<Particle>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, PionMinus)
{
  auto value = Transportation::IsApplicable<PionMinus>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, PionPlus)
{
  auto value = Transportation::IsApplicable<PionPlus>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, PionZero)
{
  auto value = Transportation::IsApplicable<PionZero>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, Positron)
{
  auto value = Transportation::IsApplicable<Positron>;
  ASSERT_TRUE(value);
}

TEST_F(TransportApplicable, Proton)
{
  auto value = Transportation::IsApplicable<Proton>;
  ASSERT_TRUE(value);
}
//
//  Test something other than a particle and make sure it is not
//  always false!
//
TEST_F(TransportApplicable, PhysicsParameters)
{
  auto value = Transportation::IsApplicable<PhysicsParameters>;
  ASSERT_FALSE(value);
}
