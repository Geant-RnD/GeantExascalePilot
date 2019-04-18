cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

################################################################################

message("Using cmake: ${CMAKE_COMMAND}...")
file(READ "VERSION" VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "[A-Za-z\n\r].*" "" VERSION_STRING "${VERSION_STRING}")

################################################################################

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0053 NEW)

include(cmake/modules/CaptureCommandLine.cmake)

project(GeantExascalePilot LANGUAGES C CXX CUDA VERSION ${VERSION_STRING})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
set(CMAKE_DIRECTORY_LABELS "GeantExascalePilot")

################################################################################

include(MacroUtilities)
include(Compilers)
include(Options)
include(GNUInstallDirs)
include(ProjectSettings)
include(BuildSettings)
include(Packages)
include(ClangFormat)
# include(Documentation)

################################################################################
#   source
################################################################################

add_subdirectory(source)
add_subdirectory(examples)
#add_subdirectory(test)

################################################################################
#   Information
################################################################################

print_features()
