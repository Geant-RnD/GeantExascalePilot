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

class TransportApplicable : public ::testing::Test
{
};

TEST_F(TransportApplicable, Electron)
{
    ASSERT_TRUE(Transportation::IsApplicable<Electron>);
}

TEST_F(TransportApplicable, Gamma) { ASSERT_TRUE(Transportation::IsApplicable<Gamma>); }

TEST_F(TransportApplicable, KaonLong)
{
    ASSERT_TRUE(Transportation::IsApplicable<KaonLong>);
}

TEST_F(TransportApplicable, KaonMinus)
{
    ASSERT_TRUE(Transportation::IsApplicable<KaonMinus>);
}

TEST_F(TransportApplicable, KaonPlus)
{
    ASSERT_TRUE(Transportation::IsApplicable<KaonPlus>);
}

TEST_F(TransportApplicable, KaonShort)
{
    ASSERT_TRUE(Transportation::IsApplicable<KaonShort>);
}

TEST_F(TransportApplicable, KaonZero)
{
    ASSERT_TRUE(Transportation::IsApplicable<KaonZero>);
}

TEST_F(TransportApplicable, Neutron)
{
    ASSERT_TRUE(Transportation::IsApplicable<Neutron>);
}

TEST_F(TransportApplicable, Particle)
{
    ASSERT_TRUE(Transportation::IsApplicable<Particle>);
}

TEST_F(TransportApplicable, PhysicsParameters)
{
    ASSERT_TRUE(Transportation::IsApplicable<PhysicsParameters>);
}

TEST_F(TransportApplicable, PionMinus)
{
    ASSERT_TRUE(Transportation::IsApplicable<PionMinus>);
}

TEST_F(TransportApplicable, PionPlus)
{
    ASSERT_TRUE(Transportation::IsApplicable<PionPlus>);
}

TEST_F(TransportApplicable, PionZero)
{
    ASSERT_TRUE(Transportation::IsApplicable<PionZero>);
}

TEST_F(TransportApplicable, Positron)
{
    ASSERT_TRUE(Transportation::IsApplicable<Positron>);
}

TEST_F(TransportApplicable, Proton) { ASSERT_TRUE(Transportation::IsApplicable<Proton>); }
