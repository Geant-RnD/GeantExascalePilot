################################################################################
#
#        Handles the global project settings
#
################################################################################

# ---------------------------------------------------------------------------- #
# compiler features
set(CMAKE_CXX_COMPILE_FEATURES
    cxx_std_11 cxx_lambdas cxx_thread_local cxx_constexpr
    cxx_decltype cxx_nullptr
    cxx_variable_templates cxx_template_template_parameters
    cxx_deleted_functions cxx_auto_type cxx_alias_templates)

# ---------------------------------------------------------------------------- #
#
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_C_STANDARD 11 CACHE STRING "C language standard")
set(CMAKE_CXX_STANDARD 14 CACHE STRING "CXX language standard")
set(CMAKE_CUDA_STANDARD 11 CACHE STRING "CUDA language standard")
set(CMAKE_C_STANDARD_REQUIRED ON CACHE BOOL "Require the C language standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the CXX language standard")
set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "Require the CUDA language standard")

set(CUDA_PROPERTIES
    CUDA_STANDARD               ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED      ON
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set(CMAKE_BUILD_TYPE Release)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _CONFIG)

add_feature(CMAKE_C_FLAGS_${_CONFIG} "C compiler build-specific flags")
add_feature(CMAKE_CXX_FLAGS_${_CONFIG} "C++ compiler build-specific flags")

# ---------------------------------------------------------------------------- #
# cmake installation folder
#
set(CMAKE_INSTALL_CONFIGDIR  ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    CACHE PATH "Installation directory for CMake package config files")

# ---------------------------------------------------------------------------- #
# create the full path version and generic path versions
#
foreach(_TYPE DATAROOT CMAKE INCLUDE LIB BIN MAN DOC)
    # set the absolute versions
    if(NOT IS_ABSOLUTE "${CMAKE_INSTALL_${_TYPE}DIR}")
        set(CMAKE_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_${_TYPE}DIR})
    else(NOT IS_ABSOLUTE "${CMAKE_INSTALL_${_TYPE}DIR}")
        set(CMAKE_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_${_TYPE}DIR})
    endif(NOT IS_ABSOLUTE "${CMAKE_INSTALL_${_TYPE}DIR}")

    # generic "PROJECT_INSTALL_" variables (used by documentation)"
    set(PROJECT_INSTALL_${_TYPE}DIR ${CMAKE_INSTALL_${TYPE}DIR})
    set(PROJECT_INSTALL_FULL_${_TYPE}DIR ${CMAKE_INSTALL_FULL_${TYPE}DIR})
endforeach()

# ---------------------------------------------------------------------------- #
# set the output directory (critical on Windows)
#
foreach(_TYPE ARCHIVE LIBRARY RUNTIME)
    # if ${PROJECT_NAME}_OUTPUT_DIR is not defined, set to CMAKE_BINARY_DIR
    if(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
        set(${PROJECT_NAME}_OUTPUT_DIR ${CMAKE_BINARY_DIR})
    endif(NOT DEFINED ${PROJECT_NAME}_OUTPUT_DIR OR "${${PROJECT_NAME}_OUTPUT_DIR}" STREQUAL "")
    # set the CMAKE_{ARCHIVE,LIBRARY,RUNTIME}_OUTPUT_DIRECTORY variables
    if(WIN32)
        # on Windows, separate types into different directories
        string(TOLOWER "${_TYPE}" _LTYPE)
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR}/outputs/${_LTYPE})
    else(WIN32)
        # on UNIX, just set to same directory
        set(CMAKE_${_TYPE}_OUTPUT_DIRECTORY ${${PROJECT_NAME}_OUTPUT_DIR})
    endif(WIN32)
endforeach(_TYPE ARCHIVE LIBRARY RUNTIME)


# ---------------------------------------------------------------------------- #
#  debug macro
#
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    list(APPEND ${PROJECT_NAME}_DEFINITIONS DEBUG)
else("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    list(APPEND ${PROJECT_NAME}_DEFINITIONS NDEBUG)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
