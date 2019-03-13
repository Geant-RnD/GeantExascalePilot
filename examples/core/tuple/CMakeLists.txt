cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Tuple LANGUAGES C CXX CUDA)

find_package(GeantExascalePilot REQUIRED)
include(MacroUtilities)

glob_files(OUTPUT_VAR headers
    DIRECTORIES source EXTENSIONS h hh hxx hpp cuh)
glob_files(OUTPUT_VAR sources
    DIRECTORIES source EXTENSIONS c cc cxx cpp cu)

set(EXE_NAME test_tuple)

CREATE_EXECUTABLE(
    TARGET_NAME             ${EXE_NAME}
    HEADERS                 ${headers}
    SOURCES                 ${CMAKE_CURRENT_LIST_DIR}/${EXE_NAME}.cpp ${sources}
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
    LINK_LIBRARIES          ${GeantExascalePilot_LIBRARIES}
)
