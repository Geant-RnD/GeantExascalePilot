cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Tuple LANGUAGES C CXX CUDA)

find_package(GeantExascalePilot REQUIRED)
find_package(TiMemory)

if(TiMemory_FOUND)
    find_package(MPI REQUIRED)
    include_directories(${TiMemory_INCLUDE_DIRS} ${MPI_C_HEADER_DIR} ${MPI_CXX_HEADER_DIR} ${MPI_CXX_ADDITIONAL_INCLUDE_DIRS})
    add_definitions(-DGEANT_USE_TIMEMORY)
endif()

include(MacroUtilities)

glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR headers
    DIRECTORIES source EXTENSIONS h hh hxx hpp cuh)
glob_files(EXCLUDE_CURRENT_DIR OUTPUT_VAR sources
    DIRECTORIES source EXTENSIONS c cc cxx cpp cu)

CREATE_EXECUTABLE(
   TARGET_NAME             test_track
   HEADERS                 ${headers}
   SOURCES                 test_track.cpp
   INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}/source
)
  

