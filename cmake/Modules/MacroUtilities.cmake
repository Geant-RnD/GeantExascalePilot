# MacroUtilities - useful macros and functions for generic tasks
#
#
# General
# --------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a  feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
#

# - Include guard
if(__${PROJECT_NAME}_macroutilities_isloaded)
  return()
endif()
set(__${PROJECT_NAME}_macroutilities_isloaded YES)

cmake_policy(PUSH)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    cmake_policy(SET CMP0054 NEW)
endif()

include(CMakeDependentOption)
include(CMakeParseArguments)


#-----------------------------------------------------------------------
# macro safe_remove_duplicates(<list>)
#       ensures remove_duplicates is only called if list has values
#
MACRO(safe_remove_duplicates _list)
    if(NOT "${${_list}}" STREQUAL "")
        list(REMOVE_DUPLICATES ${_list})
    endif(NOT "${${_list}}" STREQUAL "")
ENDMACRO()


#-----------------------------------------------------------------------
# function - capitalize - make a string capitalized (first letter is capital)
#   usage:
#       capitalize("SHARED" CShared)
#   message(STATUS "-- CShared is \"${CShared}\"")
#   $ -- CShared is "Shared"
FUNCTION(capitalize str var)
    # make string lower
    string(TOLOWER "${str}" str)
    string(SUBSTRING "${str}" 0 1 _first)
    string(TOUPPER "${_first}" _first)
    string(SUBSTRING "${str}" 1 -1 _remainder)
    string(CONCAT str "${_first}" "${_remainder}")
    set(${var} "${str}" PARENT_SCOPE)
ENDFUNCTION()


#-----------------------------------------------------------------------
# GENERAL
#-----------------------------------------------------------------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a project feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
FUNCTION(ADD_FEATURE _var _description)
  set(EXTRA_DESC "")
  foreach(currentArg ${ARGN})
      if(NOT "${currentArg}" STREQUAL "${_var}" AND
         NOT "${currentArg}" STREQUAL "${_description}")
          set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
      endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY PROJECT_FEATURES ${_var})
  #set(${_var} ${${_var}} CACHE INTERNAL "${_description}${EXTRA_DESC}")

  set_property(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")
ENDFUNCTION()


#------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE])
#          Add an option and add as a feature if NO_FEATURE is not provided
#
FUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)
    SET(_FEATURE ${ARGN})
    OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    IF(NOT "${_FEATURE}" STREQUAL "NO_FEATURE")
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ELSE()
        MARK_AS_ADVANCED(${_NAME})
    ENDIF()
ENDFUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)

#------------------------------------------------------------------------------#

FUNCTION(GLOB_FILES)
    # parse args
    cmake_parse_arguments(M
        # options
        "EXCLUDE_CURRENT_DIR"
        # single value args
        "OUTPUT_VAR"
        # multiple value args
        "DIRECTORIES;EXTENSIONS"
        ${ARGN})

    set(_FILES )
    foreach(EXT ${M_EXTENSIONS})
        foreach(DIR ${M_DIRECTORIES})
            file(GLOB TMP "${CMAKE_CURRENT_LIST_DIR}/${DIR}/*.${EXT}")
            list(APPEND _FILES ${TMP})
        endforeach()
        if(NOT M_EXCLUDE_CURRENT_DIR)
            file(GLOB TMP "${CMAKE_CURRENT_LIST_DIR}/*.${EXT}")
            list(APPEND _FILES ${TMP})
        endif()
    endforeach()

    safe_remove_duplicates(_FILES)
    set(${M_OUTPUT_VAR} ${_FILES})
    set(${M_OUTPUT_VAR} ${${M_OUTPUT_VAR}} PARENT_SCOPE)
ENDFUNCTION()


#------------------------------------------------------------------------------#
# macro for creating a library target
#
FUNCTION(CREATE_LIBRARY)

    # list of arguments taking multiple values
    set(multival_args
        HEADERS SOURCES PROPERTIES DEFINITIONS
        INCLUDE_DIRECTORIES LINK_LIBRARIES SYSTEM_INCLUDE_DIRECTORIES
        CFLAGS CXXFLAGS CUDAFLAGS INSTALL_DESTINATION)

    # parse args
    cmake_parse_arguments(LIB
        "INSTALL"                               # options
        "TARGET_NAME;OUTPUT_NAME;TYPE;PREFIX"   # single value args
        "${multival_args}"                      # multiple value args
        ${ARGN})

    # defaults
    if(NOT LIB_OUTPUT_NAME)
        string(REPLACE "::" "_" LIB_OUTPUT_NAME "${LIB_TARGET_NAME}")
    endif()

    if(NOT LIB_PREFIX)
        set(LIB_PREFIX lib)
    endif()

    if(NOT LIB_TYPE)
        if(BUILD_SHARED_LIBS)
            set(LIB_TYPE SHARED)
        else()
            set(LIB_TYPE STATIC)
        endif()
    endif()

    set(LIB_FORMAT_TARGET ${LIB_OUTPUT_NAME}-format)
    if(NOT TARGET ${LIB_FORMAT_TARGET})
        geant_format_target(
            NAME ${LIB_FORMAT_TARGET}
            SOURCES ${LIB_HEADERS} ${LIB_SOURCES})
    endif()

    # create library
    add_library(${LIB_TARGET_NAME} ${LIB_TYPE} ${LIB_SOURCES} ${LIB_HEADERS})

    # link library
    target_link_libraries(${LIB_TARGET_NAME}
        ${LIB_LINK_LIBRARIES} ${${PROJECT_NAME}_LINK_LIBRARIES})

    # include dirs
    target_include_directories(${LIB_TARGET_NAME} PRIVATE
        ${LIB_INCLUDE_DIRECTORIES} ${${PROJECT_NAME}_INCLUDE_DIRECTORIES}
        INTERFACE ${CMAKE_INSTALL_PREFIX}/include)

    # system include dirs
    target_include_directories(${LIB_TARGET_NAME} SYSTEM PRIVATE
        ${LIB_SYSTEM_INCLUDE_DIRECTORIES} ${${PROJECT_NAME}_SYSTEM_INCLUDE_DIRECTORIES})

    # link options
    if(CMAKE_VERSION VERSION_GREATER 3.13)
        target_link_options(${LIB_TARGET_NAME} PUBLIC
            ${LIB_LINK_OPTIONS} ${${PROJECT_NAME}_LINK_OPTIONS})
    endif()

    # target properties
    set_target_properties(${LIB_TARGET_NAME} PROPERTIES
        PREFIX                      "${LIB_PREFIX}"
        OUTPUT_NAME                 "${LIB_OUTPUT_NAME}"
        ${LIB_PROPERTIES}
        ${${PROJECT_NAME}_PROPERTIES})

    # compile defs
    target_compile_definitions(${LIB_TARGET_NAME} PUBLIC
        ${${PROJECT_NAME}_DEFINITIONS}
        ${LIB_DEFINITIONS})

    # compile flags
    target_compile_options(${LIB_TARGET_NAME} PUBLIC
        $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS} ${LIB_CFLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS} ${LIB_CXXFLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS} ${LIB_CUDAFLAGS}>)

    if(LIB_INSTALL AND NOT LIB_INSTALL_DESTINATION)
        set(LIB_INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()

    if(LIB_INSTALL_DESTINATION)
        # install headers
        foreach(_HEADER ${LIB_HEADERS})
            get_filename_component(HEADER_RELATIVE ${_HEADER} DIRECTORY)
            string(REPLACE "${PROJECT_SOURCE_DIR}/source/" "" HEADER_RELATIVE "${HEADER_RELATIVE}")
            install(FILES ${_HEADER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${HEADER_RELATIVE})
            #message("INSTALL: ${_HEADER} ==> ${CMAKE_INSTALL_INCLUDEDIR}/${HEADER_RELATIVE}")
        endforeach()

        # Install the compiled library
        install(TARGETS ${LIB_TARGET_NAME} DESTINATION ${LIB_INSTALL_DESTINATION}
            EXPORT ${LIB_TARGET_NAME})

        # install export
        install(EXPORT ${LIB_TARGET_NAME}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cmake/${PROJECT_NAME})

        # generate export for build tree
        export(TARGETS ${LIB_TARGET_NAME}
            FILE ${CMAKE_BINARY_DIR}/exports/${LIB_TARGET_NAME}.cmake)

        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_COMPONENTS ${LIB_TARGET_NAME})
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_${LIB_TYPE}_COMPONENTS ${LIB_TARGET_NAME})
    endif()

ENDFUNCTION()


#------------------------------------------------------------------------------#
# macro for creating a library target
#
FUNCTION(CREATE_EXECUTABLE)

    # list of arguments taking multiple values
    set(multival_args
        HEADERS SOURCES PROPERTIES DEFINITIONS
        INCLUDE_DIRECTORIES LINK_LIBRARIES SYSTEM_INCLUDE_DIRECTORIES
        CFLAGS CXXFLAGS CUDAFLAGS INSTALL_DESTINATION)

    # parse args
    cmake_parse_arguments(EXE
        "INSTALL"                               # options
        "TARGET_NAME;OUTPUT_NAME"               # single value args
        "${multival_args}"                      # multiple value args
        ${ARGN})

    set(EXE_FORMAT_TARGET ${EXE_OUTPUT_NAME}-format)
    if(NOT TARGET ${EXE_FORMAT_TARGET})
        geant_format_target(
            NAME ${EXE_FORMAT_TARGET}
            SOURCES ${EXE_HEADERS} ${EXE_SOURCES})
    endif()

    # create library
    add_executable(${EXE_TARGET_NAME} ${EXE_SOURCES} ${EXE_HEADERS})

    # link library
    target_link_libraries(${EXE_TARGET_NAME}
        ${EXE_LINK_LIBRARIES} ${${PROJECT_NAME}_LINK_LIBRARIES})

    # include dirs
    target_include_directories(${EXE_TARGET_NAME} PRIVATE
        ${EXE_INCLUDE_DIRECTORIES} ${${PROJECT_NAME}_INCLUDE_DIRECTORIES}
        INTERFACE ${CMAKE_INSTALL_PREFIX}/include)

    # system include dirs
    target_include_directories(${EXE_TARGET_NAME} SYSTEM PRIVATE
        ${EXE_SYSTEM_INCLUDE_DIRECTORIES} ${${PROJECT_NAME}_SYSTEM_INCLUDE_DIRECTORIES})

    # link options
    if(CMAKE_VERSION VERSION_GREATER 3.13)
        target_link_options(${EXE_TARGET_NAME} PUBLIC
            ${EXE_LINK_OPTIONS} ${${PROJECT_NAME}_LINK_OPTIONS})
    endif()

    # target properties
    set(_PROPERTIES ${EXE_PROPERTIES} ${${PROJECT_NAME}_PROPERTIES})
    if(NOT "${_PROPERTIES}" STREQUAL "")
        set_target_properties(${EXE_TARGET_NAME} PROPERTIES ${_PROPERTIES})
    endif()

    # compile defs
    target_compile_definitions(${EXE_TARGET_NAME} PUBLIC
        ${${PROJECT_NAME}_DEFINITIONS}
        ${EXE_DEFINITIONS})

    # compile flags
    target_compile_options(${EXE_TARGET_NAME} PUBLIC
        $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS} ${EXE_CFLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS} ${EXE_CXXFLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS} ${EXE_CUDAFLAGS}>)

    if(EXE_INSTALL AND NOT EXE_INSTALL_DESTINATION)
        set(EXE_INSTALL_DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    if(EXE_INSTALL_DESTINATION)
        # Install the exe
        install(TARGETS ${EXE_TARGET_NAME} DESTINATION ${EXE_INSTALL_DESTINATION})
    endif()

ENDFUNCTION()

#------------------------------------------------------------------------------#
# macro add_googletest()
#
# Adds a unit test and links against googletest. Additional arguments are linked
# against the test.
#
function(add_googletest _FILENAME)
    get_filename_component(_NAME "${_FILENAME}" NAME_WE)
    create_executable(TARGET_NAME ${_NAME} SOURCES ${_FILENAME})
    target_link_libraries(${_NAME} gtest gmock gtest_main ${ARGN})
    add_test(NAME ${_NAME} COMMAND ${_NAME})
endfunction()

#------------------------------------------------------------------------------#
# macro CHECKOUT_GIT_SUBMODULE()
#
#   Run "git submodule update" if a file in a submodule does not exist
#
#   ARGS:
#       RECURSIVE (option) -- add "--recursive" flag
#       RELATIVE_PATH (one value) -- typically the relative path to submodule
#                                    from PROJECT_SOURCE_DIR
#       WORKING_DIRECTORY (one value) -- (default: PROJECT_SOURCE_DIR)
#       TEST_FILE (one value) -- file to check for (default: CMakeLists.txt)
#       ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
FUNCTION(CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE"
        "ADDITIONAL_CMDS"
        ${ARGN})
    find_package(Git)
    if(NOT Git_FOUND)
        message(WARNING "Git not found. submodule ${CHECKOUT_RELATIVE_PATH} not checked out")
        return()
    endif()

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif(NOT CHECKOUT_WORKING_DIRECTORY)

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif(NOT CHECKOUT_TEST_FILE)

    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    if(NOT EXISTS "${_DIR}")
        message(FATAL_ERROR "submodule directory does not exist")
    endif(NOT EXISTS "${_DIR}")

    # if this file exists --> project has been checked out
    # if not exists --> not been checked out
    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")

    set(_RECURSE )
    if(CHECKOUT_RECURSIVE)
        set(_RECURSE --recursive)
    endif(CHECKOUT_RECURSIVE)

    # submodule typically should be updated if already checked out but this should be disabled
    # if a submodule has been updated but not staged into the commit
    add_option(GEANT_SUBMODULE_UPDATE "Run git submodule update if submodules already checked out" ON)

    # if the module has not been checked out
    if(NOT EXISTS "${_TEST_FILE}")
        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                    ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # check the return code
        if(RET GREATER 0)
            set(_CMD "${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}")
            message(STATUS "macro(CHECKOUT_SUBMODULE) failed.")
            message(WARNING "Command: \"${_CMD}\"")
            return()
        endif()
    elseif(GEANT_SUBMODULE_UPDATE)
        set(MSG "Executing '${GIT_EXECUTABLE} submodule update ${_RECURSE} ${CHECKOUT_RELATIVE_PATH}'...")
        set(MSG "${MSG} Disable with GEANT_SUBMODULE_UPDATE=OFF...")
        message(STATUS "${MSG}")
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update ${_RECURSE} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY})
    endif()

ENDFUNCTION()


#------------------------------------------------------------------------------#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
FUNCTION(print_enabled_features)
    set(_basemsg "The following features are defined/enabled (+):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(${_feature})
            # add feature to text
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            # get description
            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            # print description, if not standard ON/OFF, print what is set to
            if(_desc)
                if(NOT "${${_feature}}" STREQUAL "ON" AND
                   NOT "${${_feature}}" STREQUAL "TRUE")
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc} -- [\"${${_feature}}\"]")
                else()
                    string(REGEX REPLACE "^USE_" "" _feature_tmp "${_feature}")
                    string(TOLOWER "${_feature_tmp}" _feature_tmp_l)
                    capitalize("${_feature_tmp}" _feature_tmp_c)
                    foreach(_var _feature_tmp _feature_tmp_l _feature_tmp_c)
                        set(_ver "${${${_var}}_VERSION}")
                        if(NOT "${_ver}" STREQUAL "")
                            set(_desc "${_desc} -- [found version ${_ver}]")
                            break()
                        endif()
                        unset(_ver)
                    endforeach(_var _feature_tmp _feature_tmp_l _feature_tmp_c)
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                endif()
                set(_desc NOTFOUND)
            endif(_desc)
            # check for subfeatures
            get_property(_subfeatures GLOBAL PROPERTY ${_feature}_FEATURES)
            # remove duplicates and sort if subfeatures exist
            if(NOT "${_subfeatures}" STREQUAL "")
                list(REMOVE_DUPLICATES _subfeatures)
                list(SORT _subfeatures)
            endif()

            # sort enabled and disabled features into lists
            set(_enabled_subfeatures )
            set(_disabled_subfeatures )
            foreach(_subfeature ${_subfeatures})
                if(${_subfeature})
                    list(APPEND _enabled_subfeatures ${_subfeature})
                else()
                    list(APPEND _disabled_subfeatures ${_subfeature})
                endif()
            endforeach()

            # loop over enabled subfeatures
            foreach(_subfeature ${_enabled_subfeatures})
                # add subfeature to text
                set(_currentFeatureText "${_currentFeatureText}\n       + ${_subfeature}")
                # get subfeature description
                get_property(_subdesc GLOBAL PROPERTY ${_feature}_${_subfeature}_DESCRIPTION)
                # print subfeature description. If not standard ON/OFF, print
                # what is set to
                if(_subdesc)
                    if(NOT "${${_subfeature}}" STREQUAL "ON" AND
                       NOT "${${_subfeature}}" STREQUAL "TRUE")
                        set(_currentFeatureText "${_currentFeatureText}: ${_subdesc} -- [\"${${_subfeature}}\"]")
                    else()
                        set(_currentFeatureText "${_currentFeatureText}: ${_subdesc}")
                    endif()
                    set(_subdesc NOTFOUND)
                endif(_subdesc)
            endforeach(_subfeature)

            # loop over disabled subfeatures
            foreach(_subfeature ${_disabled_subfeatures})
                # add subfeature to text
                set(_currentFeatureText "${_currentFeatureText}\n       - ${_subfeature}")
                # get subfeature description
                get_property(_subdesc GLOBAL PROPERTY ${_feature}_${_subfeature}_DESCRIPTION)
                # print subfeature description.
                if(_subdesc)
                    set(_currentFeatureText "${_currentFeatureText}: ${_subdesc}")
                    set(_subdesc NOTFOUND)
                endif(_subdesc)
            endforeach(_subfeature)

        endif(${_feature})
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()


#------------------------------------------------------------------------------#
# function print_disabled_features()
#          Print disabled features plus their docstrings.
#
FUNCTION(print_disabled_features)
    set(_basemsg "The following features are NOT defined/enabled (-):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(NOT ${_feature})
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")

            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)

            if(_desc)
              set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
              set(_desc NOTFOUND)
            endif(_desc)
        endif()
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()

#------------------------------------------------------------------------------#
# function print_features()
#          Print all features plus their docstrings.
#
FUNCTION(print_features)
    message(STATUS "")
    print_enabled_features()
    print_disabled_features()
ENDFUNCTION()

cmake_policy(POP)
