cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Tuple LANGUAGES C CXX CUDA)

find_package(GeantExascalePilot REQUIRED)
include(MacroUtilities)

glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR headers
    DIRECTORIES source EXTENSIONS h hh hxx hpp cuh)
glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR sources
    DIRECTORIES source EXTENSIONS c cc cxx cpp cu)

CREATE_EXECUTABLE(
    TARGET_NAME             test_tuple
    HEADERS                 ${headers}
    SOURCES                 test_tuple.cpp ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          ${GeantExascalePilot_LIBRARIES}
)

CREATE_EXECUTABLE(
    TARGET_NAME             bench_tuple
    HEADERS                 ${headers}
    SOURCES                 bench_tuple.cpp ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          ${GeantExascalePilot_LIBRARIES}
)
