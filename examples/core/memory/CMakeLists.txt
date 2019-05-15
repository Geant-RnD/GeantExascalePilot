cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Memory LANGUAGES C CXX CUDA)

find_package(GeantExascalePilot REQUIRED)
find_package(CUDA REQUIRED)

include(MacroUtilities)

glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR headers
    DIRECTORIES source EXTENSIONS h hh hxx hpp cuh)
glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR sources
    DIRECTORIES source EXTENSIONS c cc cxx cpp cu)

CREATE_EXECUTABLE(
    TARGET_NAME             test_memory
    HEADERS                 ${headers}
    SOURCES                 test_memory.cpp ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          GeantExascalePilot ${CUDA_LIBRARIES}
)
