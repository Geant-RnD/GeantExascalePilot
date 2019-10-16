
include(MacroUtilities)

glob_files(OUTPUT_VAR headers EXTENSIONS h hh hxx hpp cuh)
glob_files(OUTPUT_VAR sources
    DIRECTORIES src EXTENSIONS c cc cxx cpp cu)

# name the library after the folder
get_filename_component(LIBNAME ${CMAKE_CURRENT_LIST_DIR} NAME)

# build a static library
if(BUILD_STATIC_LIBS)
    CREATE_LIBRARY(
        INSTALL
        TARGET_NAME             ${LIBNAME}-static
        OUTPUT_NAME             ${LIBNAME}
        TYPE                    STATIC
        HEADERS                 ${headers}
        SOURCES                 ${sources}
        INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}
    )
    set(LINK_TARGET ${LIBNAME}-static)
endif()

# build a shared library
if(BUILD_SHARED_LIBS)
    CREATE_LIBRARY(
        INSTALL
        TARGET_NAME             ${LIBNAME}-shared
        OUTPUT_NAME             ${LIBNAME}
        TYPE                    SHARED
        HEADERS                 ${headers}
        SOURCES                 ${sources}
        INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}
        PROPERTIES              VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    set(LINK_TARGET ${LIBNAME}-shared)
endif()

add_subdirectory(test)
