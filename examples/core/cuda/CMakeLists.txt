cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Tuple LANGUAGES C CXX CUDA)

find_package(GeantExascalePilot REQUIRED)
find_package(VecGeom REQUIRED)
find_package(VecCore REQUIRED)

include(MacroUtilities)

CREATE_EXECUTABLE(
    TARGET_NAME             test_cuda_map
    HEADERS                 ${headers}
    SOURCES                 MapTest.cpp MapTest.cu
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR} ${VECGEOM_INCLUDE_DIRS} ${VecCore_INCLUDE_DIRS}
    LINK_LIBRARIES          ${GeantExascalePilot_LIBRARIES} ${VECGEOM_LIBRARIES}
)

CREATE_EXECUTABLE(
    TARGET_NAME             test_cuda_map_class
    HEADERS                 ${headers}
    SOURCES                 MapTestClass.cpp MapTestClass.cu
    INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR} ${VECGEOM_INCLUDE_DIRS} ${VecCore_INCLUDE_DIRS}
    LINK_LIBRARIES          ${GeantExascalePilot_LIBRARIES} ${VECGEOM_LIBRARIES}
)
