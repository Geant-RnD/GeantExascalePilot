
include(MacroUtilities)

glob_files(OUTPUT_VAR headers
    EXTENSIONS h hh hxx hpp cuh
)

glob_files(OUTPUT_VAR sources
    DIRECTORIES src
    EXTENSIONS c cc cxx cpp cu
)

set(LIBNAME magneticfield)
if(BUILD_SHARED_LIBS)
    CREATE_LIBRARY(
        TARGET_NAME             ${LIBNAME}-shared
        OUTPUT_NAME             ${LIBNAME}
        TYPE                    SHARED
        SOURCES                 ${headers} ${sources}
        INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}
        PROPERTIES              VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

if(BUILD_STATIC_LIBS)
    CREATE_LIBRARY(
        TARGET_NAME             ${LIBNAME}-static
        OUTPUT_NAME             ${LIBNAME}
        TYPE                    STATIC
        SOURCES                 ${headers} ${sources}
        INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}
    )
endif()
