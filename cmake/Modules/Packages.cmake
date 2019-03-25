################################################################################
#
#        Handles the external packages
#
################################################################################

include(FindPackageHandleStandardArgs)

################################################################################
#
#                               Threading
#
################################################################################

if(CMAKE_C_COMPILER_IS_INTEL OR CMAKE_CXX_COMPILER_IS_INTEL)
    if(NOT WIN32)
        set(CMAKE_THREAD_PREFER_PTHREAD ON)
        set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "Use -pthread vs. -lpthread" FORCE)
    endif()

    find_package(Threads)
    if(Threads_FOUND)
        list(APPEND EXTERNAL_LIBRARIES Threads::Threads)
    endif()
endif()


################################################################################
#
#        GCov
#
################################################################################

if(GEANT_USE_COVERAGE)
    find_library(GCOV_LIBRARY gcov)
    if(GCOV_LIBRARY)
        list(APPEND EXTERNAL_LIBRARIES ${GCOV_LIBRARY})
    else()
        list(APPEND EXTERNAL_LIBRARIES gcov)
    endif()
    add(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
endif()


################################################################################
#
#                               TiMemory
#
################################################################################

if(GEANT_USE_TIMEMORY)
    find_package(TiMemory)

    if(TiMemory_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TiMemory_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES
            ${TiMemory_LIBRARIES} ${TiMemory_C_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS GEANT_USE_TIMEMORY)
    endif()

endif()


################################################################################
#
#        Google PerfTools
#
################################################################################

if(GEANT_USE_GPERF)
    find_package(GPerfTools COMPONENTS profiler)

    if(GPerfTools_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${GPerfTools_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${GPerfTools_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS GEANT_USE_GPERF)
    endif()

endif()


################################################################################
#
#        CUDA
#
################################################################################

add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA NVCC compiler flags")
add_feature(CUDA_ARCH "CUDA architecture (e.g. '35' means '-arch=sm_35')")
set(CUDA_ARCH "62" CACHE STRING "CUDA architecture flag")

if(GEANT_USE_NVTX)
    find_library(NVTX_LIBRARY
        NAMES nvToolsExt
        PATHS /usr/local/cuda
        HINTS /usr/local/cuda
        PATH_SUFFIXES lib lib64)
else()
    unset(NVTX_LIBRARY CACHE)
endif()

if(NVTX_LIBRARY)
    list(APPEND EXTERNAL_CUDA_LIBRARIES ${NVTX_LIBRARY})
    list(APPEND ${PROJECT_NAME}_DEFINITIONS GEANT_USE_NVTX)
endif()

list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
    -arch=sm_${CUDA_ARCH}
    --default-stream per-thread
    --compiler-bindir=${CMAKE_CXX_COMPILER})

list(APPEND EXTERNAL_LIBRARIES ${CUDA_npp_LIBRARY})
list(APPEND EXTERNAL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


################################################################################
#
#        ITTNOTIFY (for VTune)
#
################################################################################
if(GEANT_USE_ITTNOTIFY)
    find_package(ittnotify)

    if(ittnotify_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${ITTNOTIFY_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${ITTNOTIFY_LIBRARIES})
    else()
        message(WARNING "ittnotify not found. Set \"VTUNE_AMPLIFIER_201{7,8,9}_DIR\" or \"VTUNE_AMPLIFIER_XE_201{7,8,9}_DIR\" in environment")
    endif()
endif()


################################################################################
#
#        External variables
#
################################################################################

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
foreach(_DIR ${EXTERNAL_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach()

# include dirs
set(${PROJECT_NAME}_INCLUDE_DIRECTORIES
    ${EXTERNAL_INCLUDE_DIRS})

# link libs
set(${PROJECT_NAME}_LINK_LIBRARIES
    ${EXTERNAL_LIBRARIES})

set(${PROJECT_NAME}_PROPERTIES
    C_STANDARD                  ${CMAKE_C_STANDARD}
    C_STANDARD_REQUIRED         ${CMAKE_C_STANDARD_REQUIRED}
    CXX_STANDARD                ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED       ${CMAKE_CXX_STANDARD_REQUIRED}
    ${CUDA_PROPERTIES}
)