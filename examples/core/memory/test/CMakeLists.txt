cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(MemoryTest LANGUAGES C CXX CUDA)

find_package(GeantExascalePilot REQUIRED)

include(MacroUtilities)

glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR headers
    DIRECTORIES source EXTENSIONS h hh hxx hpp cuh)
glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR sources
    DIRECTORIES source EXTENSIONS c cc cxx cpp cu)

CREATE_EXECUTABLE(
    TARGET_NAME             bench_memory
    HEADERS                 ${headers}
    SOURCES                 bench_memory.cpp ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          GeantExascalePilot geantx-timemory
    PROPERTIES
        LINKER_LANGUAGE     CUDA
)
