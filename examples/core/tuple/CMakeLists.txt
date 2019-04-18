cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Tuple LANGUAGES C CXX CUDA)

find_package(GeantExascalePilot REQUIRED)

include(MacroUtilities)

glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR headers
    DIRECTORIES source EXTENSIONS h hh hxx hpp cuh)
glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR sources
    DIRECTORIES source EXTENSIONS c cc cxx cpp cu)

CREATE_LIBRARY(
    TARGET_NAME             example_core_tuple
    HEADERS                 ${headers}
    SOURCES                 ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          GeantExascalePilot ${TiMemory_LIBRARIES}
)

CREATE_EXECUTABLE(
    TARGET_NAME             test_tuple
    HEADERS                 ${headers}
    SOURCES                 test_tuple.cpp
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          example_core_tuple
)

CREATE_EXECUTABLE(
    TARGET_NAME             bench_tuple
    HEADERS                 ${headers}
    SOURCES                 bench_tuple.cpp
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          example_core_tuple
)

set_source_files_properties(bench_nvstd.cpp PROPERTIES LANGUAGE CUDA)

CREATE_EXECUTABLE(
    TARGET_NAME             bench_nvstd
    HEADERS                 ${headers}
    SOURCES                 bench_nvstd.cpp
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          example_core_tuple
)

CREATE_EXECUTABLE(
    TARGET_NAME             track_manager_tuple
    HEADERS                 ${headers}
    SOURCES                 track_manager_tuple.cpp
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
)
