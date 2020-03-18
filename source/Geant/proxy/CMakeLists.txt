
include(MacroUtilities)

glob_files(OUTPUT_VAR headers EXTENSIONS hpp cuh)
glob_files(OUTPUT_VAR sources
    DIRECTORIES src EXTENSIONS cpp cu)

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
        LINK_LIBRARIES          core random geometry material particles processes magneticfield
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
        LINK_LIBRARIES          core random geometry material particles processes magneticfield
    )
    set(LINK_TARGET ${LIBNAME}-shared)
endif()

add_subdirectory(test)
