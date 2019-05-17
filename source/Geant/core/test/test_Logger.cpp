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
 * @brief Test of Logger
 */
//===----------------------------------------------------------------------===//

#include "../Logger.hpp"

#include "gtest/gtest.h"
#include "Geant/core/Assert.hpp"

using namespace geantx;
using std::cout;
using std::endl;

class LoggerTest : public ::testing::Test
{
    void SetUp()
    {
        Logger& logger_instance = Logger::GetInstance();
        logger_instance.SetGlobalLevel(kPrint);
        logger_instance.SetLocalLevel(kDiagnostic);
        logger_instance.Set("screen", &std::cerr, kPrint);
        logger_instance.Remove("file");
    }
};

struct Expensive
{
    int a;
    Expensive(int start_a) : a(start_a) { REQUIRE(start_a >= 0); }
};

std::ostream& operator<<(std::ostream& os, const Expensive& rhs)
{
    cout << "--- Calling expensive ostream function ---" << endl;
    for (int i = 0; i < rhs.a; ++i)
    {
        os << i;
    }
    return os;
}

TEST_F(LoggerTest, general)
{
    LogMaster(kWarning) << "Cross sections for XYZ are missing";
    LogMaster(kInfo) << "Information!";
    LogMaster(kFatal) << "The program is about to crash!";

    Log(kError) << "A particle died a horrible death!";
    Log() << "Thread 3 transported 10234 tracks";

    // Change the level
    Logger::GetInstance().SetLocalLevel(kWarning);
    Log(kInfo) << "This shouldn't show up in stderr!";
    LogMaster(kInfo) << "This should show up!";
    Log(kPrint) << "This shouldn't show up in stderr!";
    Log(kDiagnostic) << "This diagnostic should show up in stderr!";

    // Manipulators require special workarounds in the code
    Log() << endl << "^ this was a blank line";

    // Debug statements shouldn't have to call expensive functions
    Log(kPrint) << "This expensive print statement... "
                        << Expensive(20)
                        << "... shouldn't have been called.";


    LogMaster(kPrint) << "Called an expensive statement... "
                      << Expensive(10);
}

TEST_F(LoggerTest, setting)
{
    // The "screen" output destination and minimum level can be changed at
    // will.
    LogMaster(kPrint) << "This should show up";
    Logger::GetInstance().Set("screen", &std::cerr, kWarning);

    LogMaster(kStatus) << "This should NOT show up!";
}

TEST_F(LoggerTest, logging)
{
    auto stream_sp = std::make_shared<std::ostringstream>();

    // Additional loggers can be set with different "minimum" log levels.
    // For example, you could set an ofstream on node zero that only writes
    // warnings and higher.
    Logger::GetInstance().Set("file", stream_sp, kWarning);

    LogMaster(kPrint) << "Screen only";
    LogMaster(kWarning) << "Screen and file";
    LogMaster(kError) << "Screen and file";
    LogMaster(kStatus) << "Screen only";

    const std::string& log_stream = stream_sp->str();
    if (GetThisThreadID() == 0)
    {
        EXPECT_EQ("*** Screen and file\n!!! Screen and file\n", log_stream);
    }
    else
    {
        EXPECT_EQ("", log_stream);
    }
}
