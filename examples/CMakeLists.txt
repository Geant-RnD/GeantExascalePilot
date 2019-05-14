
set(${CMAKE_PROJECT_NAME}_DIR ${CMAKE_BINARY_DIR})

if(NOT GEANT_BUILD_EXAMPLES)
    return()
endif()

add_subdirectory(core)
add_subdirectory(track)
add_subdirectory(geometry)
