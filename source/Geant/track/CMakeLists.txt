
# add_library(track-static INTERFACE)
# add_library(track-shared INTERFACE)

if(GEANT_BUILD_TESTS)
    add_subdirectory(test)
endif()
