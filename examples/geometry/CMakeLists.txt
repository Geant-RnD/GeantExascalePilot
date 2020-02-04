cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

#project(Tuple LANGUAGES C CXX CUDA)
project(GeometryTests LANGUAGES CXX)

find_package(GeantExascalePilot REQUIRED)
find_package(VecGeom REQUIRED)

include(MacroUtilities)

glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR headers
    DIRECTORIES source EXTENSIONS h hh hxx hpp cuh)

glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR sources
    DIRECTORIES source EXTENSIONS c cc cxx cpp cu)

CREATE_LIBRARY(
    TARGET_NAME             example_TestEm3
    HEADERS                 ${headers}
    SOURCES                 ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          GeantExascalePilot
)

CREATE_EXECUTABLE(
    TARGET_NAME             GeometryTest
    HEADERS                 ${headers}
    SOURCES                 GeometryTest.cpp
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          example_TestEm3 GeantExascalePilot
)

CREATE_EXECUTABLE(
    TARGET_NAME             NavigationBenchmark
    HEADERS                 ${headers}
    SOURCES                 NavigationBenchmark.cpp
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          example_TestEm3 GeantExascalePilot
)
