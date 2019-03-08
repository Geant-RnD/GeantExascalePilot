################################################################################
#                                                                              #
#                               General                                        #
#                                                                              #
################################################################################

add_feature(VECGEOM_VECTOR "VecGeom vector instruction set")
add_feature(VECGEOM_BACKEND "VecGeom backend")

# Submodule options
set(CUDA ON CACHE BOOL "Enable CUDA" FORCE)
set(VC OFF CACHE BOOL "Enable VC in VecCore" FORCE)
set(BUILTIN_VECCORE OFF CACHE BOOL "VecGeom builtin" FORCE)
set(VECGEOM_VECTOR "avx" CACHE STRING "VecGeom Vector instructions")
set(VECGEOM_BACKEND "cuda" CACHE STRING "VecGeom backend")
set(CTEST OFF CACHE BOOL "VecGeom testing" FORCE)
set(VALIDATION OFF CACHE BOOL "VecGeom validation" FORCE)
set(CUDA_VOLUME_SPECIALIZATION OFF CACHE BOOL "VecGeom" FORCE)
set(NO_SPECIALIZATION ON CACHE BOOL "VecGeom" FORCE)
set(ROOT OFF CACHE BOOL "VecGeom" FORCE)
set(PTL_USE_TBB OFF CACHE BOOL "Enable TBB backend for PTL")
set(PTL_USE_ARCH ${GEANT_USE_ARCH} CACHE BOOL "" FORCE)
set(PTL_USE_GPERF ${GEANT_USE_GPERF} CACHE BOOL "" FORCE)
set(PTL_USE_SANITIZER ${GEANT_USE_SANITIZER} CACHE BOOL "" FORCE)
if(GEANT4_USE_ARCH)
    set(PTL_USE_AVX512 ${GEANT_USE_AVX512} CACHE BOOL "" FORCE)
endif()

# if not specified to only use submodules find the packages
if(NOT GEANT_USE_SUBMODULES)
    find_package(VecCore)
    find_package(VecMath)
    find_package(VecGeom)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${VecCore_INCLUDE_DIRS} ${VecMath_INCLUDE_DIRS} ${VecGeom_INCLUDE_DIR})
endif()

list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR})

# PTL submodule
checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
    RELATIVE_PATH source/PTL WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
# add submodule
add_subdirectory(PTL)
list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source)

# VecCore submodule
if(NOT VecCore_FOUND)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/VecCore WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(VecCore)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/VecCore/include)
endif()

# VecMath package or submodule
if(NOT VecMath_FOUND)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/VecMath WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(VecMath)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/VecMath/include)
endif()

# VecGeom submodule
if(NOT VecGeom_FOUND)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/VecGeom WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(VecGeom)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/VecGeom)
endif()

# append submodule configs
list(APPEND ${PROJECT_NAME}_LINK_LIBRARIES
    ptl${GEANT_LIBTARGET_EXT}
    vecgeom vecgeomcuda)

list(APPEND EXTERNAL_LIBRARIES
    ptl${GEANT_LIBTARGET_EXT}
    vecgeom vecgeomcuda)

list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source
    ${CMAKE_CURRENT_LIST_DIR}/VecCore/include
    ${CMAKE_CURRENT_LIST_DIR}/VecMath/include
    ${CMAKE_CURRENT_LIST_DIR}/VecGeom
    ${CMAKE_CURRENT_BINARY_DIR}/VecCore/include)

# finally, add Geant
add_subdirectory(Geant)
