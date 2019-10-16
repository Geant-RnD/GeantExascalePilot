
add_geant_google_test(TestNullLoggerStatement
    DISCOVER_TESTS
    SOURCES         test_NullLoggerStatement.cpp
    LINK_LIBRARIES  core${GEANT_LIBTARGET_EXT})

add_geant_google_test(TestLogger
    DISCOVER_TESTS
    SOURCES         test_Logger.cpp
    LINK_LIBRARIES  core)
