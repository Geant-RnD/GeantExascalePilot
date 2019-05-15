
include(MacroUtilities)

# header directories
set(HEADER_DIRS
    management
)

# source directories
set(SOURCE_DIRS )
foreach(_DIR ${HEADER_DIRS})
    set(_FULLDIR ${CMAKE_CURRENT_LIST_DIR}/${_DIR}/src)
    if(EXISTS ${_FULLDIR} AND IS_DIRECTORY ${_FULLDIR})
        list(APPEND SOURCE_DIRS ${_DIR}/src)
    endif()
endforeach()

glob_files(OUTPUT_VAR headers DIRECTORIES ${HEADER_DIRS}
    EXTENSIONS h hh hxx hpp cuh)

glob_files(OUTPUT_VAR sources DIRECTORIES ${SOURCE_DIRS}
    EXTENSIONS c cc cxx cpp cu)

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
        LINK_LIBRARIES          core-static ${VECGEOM_LIBRARIES}
        INCLUDE_DIRECTORIES     ${CMAKE_CURRENT_LIST_DIR}
        PROPERTIES              LINKER_LANGUAGE CUDA
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
        LINK_LIBRARIES          core-shared ${VECGEOM_LIBRARIES}
        PROPERTIES              VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
                                LINKER_LANGUAGE CUDA
    )
    set(LINK_TARGET ${LIBNAME}-shared)
endif()
