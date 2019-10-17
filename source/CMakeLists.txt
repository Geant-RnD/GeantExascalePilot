##########################################################################################
#                                                                                        #
#                                         General                                        #
#                                                                                        #
##########################################################################################

add_feature(VECGEOM_VECTOR "VecGeom vector instruction set")
add_feature(VECGEOM_BACKEND "VecGeom backend")

# Submodule options
set(CUDA ON CACHE BOOL "Enable CUDA" FORCE)
set(VC OFF CACHE BOOL "Enable VC in VecCore" FORCE)
set(BUILTIN_VECCORE OFF CACHE BOOL "VecGeom builtin" FORCE)
set(VECGEOM_VECTOR "avx" CACHE STRING "VecGeom Vector instructions")
set(VECGEOM_BACKEND "Scalar" CACHE STRING "VecGeom backend")
set(VecCore_COMPONENTS ${VecCore_COMPONENTS} "CUDA")
if(NOT "${VECGEOM_BACKEND}" MATCHES "Scalar")
   set(VecCore_COMPONENTS ${VecCore_COMPONENTS} ${VECGEOM_BACKEND})
endif()
set(CTEST OFF CACHE BOOL "VecGeom testing" FORCE)
set(VALIDATION OFF CACHE BOOL "VecGeom validation" FORCE)
set(CUDA_VOLUME_SPECIALIZATION OFF CACHE BOOL "VecGeom" FORCE)
set(NO_SPECIALIZATION ON CACHE BOOL "VecGeom" FORCE)
set(ROOT OFF CACHE BOOL "VecGeom" FORCE)
set(PTL_USE_TBB OFF CACHE BOOL "Enable TBB backend for PTL")
set(PTL_USE_ARCH ${GEANT_USE_ARCH} CACHE BOOL "" FORCE)
set(PTL_USE_GPERF ${GEANT_USE_GPERF} CACHE BOOL "" FORCE)
set(PTL_USE_SANITIZER ${GEANT_USE_SANITIZER} CACHE BOOL "" FORCE)
set(TIMEMORY_BUILD_C OFF CACHE BOOL "Build TiMemory C interface" FORCE)
set(TIMEMORY_BUILD_PYTHON OFF CACHE BOOL "Build TiMemory Python interface")
set(TIMEMORY_SKIP_BUILD ON CACHE BOOL "Do not build the timemory C/C++/Python libraries")
if(GEANT4_USE_ARCH)
    set(PTL_USE_AVX512 ${GEANT_USE_AVX512} CACHE BOOL "" FORCE)
endif()

add_library(geant-headers INTERFACE)
target_include_directories(geant-headers INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

# if not specified to only use submodules find the packages
if(NOT GEANT_USE_SUBMODULES)
    find_package(VecCore ${VecCore_VERSION} REQUIRED COMPONENTS ${VecCore_COMPONENTS})
    find_package(VecMath)
    find_package(VecGeom)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${VECCORE_INCLUDE_DIRS} ${VECMATH_INCLUDE_DIRS} ${VECGEOM_INCLUDE_DIRS})
    list(APPEND ${PROJECT_NAME}_DEFINITIONS
        ${VECCORE_DEFINITIONS} ${VECGEOM_DEFINITIONS})
else()
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/VecCore/include
        ${CMAKE_CURRENT_LIST_DIR}/VecMath/include
        ${CMAKE_CURRENT_LIST_DIR}/VecGeom
        ${CMAKE_CURRENT_BINARY_DIR}/VecCore/include)
endif()

list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR})

#----------------------------------------------------------------------------------------#
# PTL submodule
#
checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
    RELATIVE_PATH source/PTL WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
add_subdirectory(PTL)
list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source)

#----------------------------------------------------------------------------------------#
# TiMemory submodule
#
checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
    RELATIVE_PATH source/timemory WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
add_subdirectory(timemory)
set(timemory_FIND_COMPONENTS_INTERFACE geantx-timemory)
set(timemory_DIR ${CMAKE_BINARY_DIR})
find_package(timemory COMPONENTS caliper gotcha arch papi gperftools-cpu cuda nvtx cudart)
set(TIMEMORY_LIBRARY geantx-timemory)

#----------------------------------------------------------------------------------------#
# GoogleTest submodule
#
if(GEANT_BUILD_TESTS)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/GoogleTest WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(GoogleTest)
    enable_testing()

    mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
    )
endif()

#----------------------------------------------------------------------------------------#
# VecCore submodule
#
if(NOT VecCore_FOUND)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/VecCore WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(VecCore)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/VecCore/include)
endif()

#----------------------------------------------------------------------------------------#
# VecMath submodule
#
if(NOT VecMath_FOUND)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/VecMath WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(VecMath)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/VecMath/include)
endif()

#----------------------------------------------------------------------------------------#
# VecGeom submodule
#
if(NOT VecGeom_FOUND)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/VecGeom WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(VecGeom)
    list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}/VecGeom)
endif()

# append submodule configs
if(GEANT_USE_SUBMODULES)
    set(VECGEOM_LIBRARIES vecgeom vecgeomcuda)
endif()

list(APPEND ${PROJECT_NAME}_LINK_LIBRARIES
    ptl${GEANT_LIBTARGET_EXT} ${VECGEOM_LIBRARIES} ${TIMEMORY_LIBRARY})

list(APPEND EXTERNAL_LIBRARIES
    ptl${GEANT_LIBTARGET_EXT} ${VECGEOM_LIBRARIES} ${TIMEMORY_LIBRARY})

list(APPEND ${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR} # hack until we fix Config.h
    ${CMAKE_CURRENT_LIST_DIR}/PTL/source)

#
target_include_directories(geant-headers INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_INCLUDE_DIRECTORIES}>
    $<INSTALL_INTERFACE:${${PROJECT_NAME}_INCLUDE_DIRECTORIES}>)

target_link_libraries(geant-headers INTERFACE ${EXTERNAL_LIBRARIES})

# finally, add Geant
add_subdirectory(Geant)

# get global properties
get_property(${PROJECT_NAME}_SHARED_COMPONENTS GLOBAL PROPERTY ${PROJECT_NAME}_SHARED_COMPONENTS)
get_property(${PROJECT_NAME}_STATIC_COMPONENTS GLOBAL PROPERTY ${PROJECT_NAME}_STATIC_COMPONENTS)
set(${PROJECT_NAME}_COMPONENTS ${${PROJECT_NAME}_SHARED_COMPONENTS} ${${PROJECT_NAME}_STATIC_COMPONENTS})

# configure
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/GeantConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in
    INSTALL_DESTINATION share/cmake/${PROJECT_NAME}
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

# configure twice
configure_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

if(BUILD_SHARED_LIBS)
    # create interface library
    add_library(Geant-shared INTERFACE)
    # link to components
    target_link_libraries(Geant-shared INTERFACE ${${PROJECT_NAME}_SHARED_COMPONENTS})
    # install libs
    list(APPEND INSTALL_LIBS Geant-shared)

    # create interface library
    add_library(Geant INTERFACE)
    # link to components
    target_link_libraries(Geant INTERFACE ${${PROJECT_NAME}_SHARED_COMPONENTS})
    # install libs
    list(APPEND INSTALL_LIBS Geant)
endif()

if(BUILD_STATIC_LIBS)
    # create interface library
    add_library(Geant-static INTERFACE)
    # link to components
    target_link_libraries(Geant-static INTERFACE ${${PROJECT_NAME}_STATIC_COMPONENTS})
    # install libs
    list(APPEND INSTALL_LIBS Geant-static)
endif()

# Install the interface library
install(TARGETS ${INSTALL_LIBS}
    DESTINATION ${LIB_INSTALL_DESTINATION}
    EXPORT ${PROJECT_NAME}LibraryDepends)

# install export
install(EXPORT ${PROJECT_NAME}LibraryDepends
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cmake/${PROJECT_NAME})

# generate export for build tree
#export(TARGETS ${INSTALL_LIBS}
#    FILE ${CMAKE_BINARY_DIR}/exports/${PROJECT_NAME}LibraryDepends.cmake)
