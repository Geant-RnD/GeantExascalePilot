################################################################################
#
#        Handles the CMake options
#
################################################################################

include(MacroUtilities)
include(Compilers)

if(CMAKE_C_COMPILER_IS_PGI)
    set(OpenMP_C_IMPL "=nonuma" CACHE STRING "OpenMP C library setting")
endif()

if(CMAKE_CXX_COMPILER_IS_PGI)
    set(OpenMP_CXX_IMPL "=nonuma" CACHE STRING "OpenMP C++ library setting")
    set(_USE_PTL OFF)
    set(_USE_PYBIND OFF)
endif()

# features
add_feature(CMAKE_BUILD_TYPE "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)")
add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
add_feature(${PROJECT_NAME}_C_FLAGS "C compiler flags")
add_feature(${PROJECT_NAME}_CXX_FLAGS "C++ compiler flags")
add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA compiler flags")
add_feature(CMAKE_C_STANDARD "C languae standard")
add_feature(CMAKE_CXX_STANDARD "C++11 STL standard")

# options (always available)
add_option(GEANT_USE_ARCH "Enable architecture specific flags" OFF)
add_option(GEANT_USE_SANITIZER "Enable sanitizer" OFF)
add_option(GEANT_USE_COVERAGE "Enable code coverage" OFF)
add_option(GEANT_USE_CLANG_TIDY "Enable running clang-tidy" OFF)
add_option(GEANT_USE_SUBMODULES "Use git submodules instead of find_package" OFF)
add_option(GEANT_BUILD_EXAMPLES "Build the examples" ON)
add_option(GEANT_BUILD_TESTS "Build unit tests" ON)
add_option(BUILD_STATIC_LIBS "Build static libraries" OFF)
add_option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(GEANT_USE_SANITIZER)
    set(SANITIZER_TYPE leak CACHE STRING "Type of sanitizer")
    add_feature(SANITIZER_TYPE "Type of sanitizer (-fsanitize=${SANITIZER_TYPE})")
endif()

if(BUILD_SHARED_LIBS)
    set(GEANT_LIBTARGET_EXT -shared)
else()
    set(GEANT_LIBTARGET_EXT -static)
endif()

# RPATH settings
set(_RPATH_LINK OFF)
if(APPLE)
    set(_RPATH_LINK ON)
endif()
add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Hardcode installation rpath based on link path" ${_RPATH_LINK})
unset(_RPATH_LINK)

# clang-tidy
if(GEANT_USE_CLANG_TIDY)
    find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
    add_feature(CLANG_TIDY_COMMAND "Path to clang-tidy command")
    if(NOT CLANG_TIDY_COMMAND)
        message(WARNING "GEANT_USE_CLANG_TIDY is ON but clang-tidy is not found!")
        set(GEANT_USE_CLANG_TIDY OFF)
    else()
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")

        # Create a preprocessor definition that depends on .clang-tidy content so
        # the compile command will change when .clang-tidy changes.  This ensures
        # that a subsequent build re-runs clang-tidy on all sources even if they
        # do not otherwise need to be recompiled.  Nothing actually uses this
        # definition.  We add it to targets on which we run clang-tidy just to
        # get the build dependency on the .clang-tidy file.
        file(SHA1 ${PROJECT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
        set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
        unset(clang_tidy_sha1)
    endif()
endif()

configure_file(${PROJECT_SOURCE_DIR}/.clang-tidy ${PROJECT_SOURCE_DIR}/.clang-tidy COPYONLY)
