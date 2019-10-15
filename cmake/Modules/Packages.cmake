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
set(CUDA_AUTO_ARCH "auto")
set(CUDA_ARCHITECTURES auto pascal volta turing)
set(CUDA_ARCH "${CUDA_AUTO_ARCH}" CACHE STRING
    "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
add_feature(CUDA_ARCH "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS ${CUDA_ARCHITECTURES})

set(cuda_pascal_arch    60)
set(cuda_volta_arch     70)
set(cuda_turing_arch    75)

if(NOT "${CUDA_ARCH}" STREQUAL "${CUDA_AUTO_ARCH}")
    if(NOT "${CUDA_ARCH}" IN_LIST CUDA_ARCHITECTURES)
        message(WARNING
            "CUDA architecture \"${CUDA_ARCH}\" not known. Options: ${CUDA_ARCH}")
        unset(CUDA_ARCH CACHE)
        set(CUDA_ARCH "${CUDA_AUTO_ARCH}")
    else()
        set(_ARCH_NUM ${cuda_${CUDA_ARCH}_arch})
    endif()
endif()

add_library(geantx-cuda INTERFACE)
list(APPEND EXTERNAL_CUDA_LIBRARIES geantx-cuda)

if(CUDA_MAJOR_VERSION VERSION_GREATER 10 OR CUDA_MAJOR_VERSION MATCHES 10)
    target_compile_options(geantx-cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
        $<IF:$<STREQUAL:"${CUDA_ARCH}","${CUDA_AUTO_ARCH}">,-arch=sm_60,-arch=sm_${_ARCH_NUM}>
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_75,code=compute_75
        >)
else()
    target_compile_options(geantx-cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
        $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_AUTO_ARCH}>,-arch=sm_60,-arch=sm_${_ARCH_NUM}>
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_70,code=compute_70
        >)
endif()

add_feature(CUDA_ARCH "CUDA architecture")
#   30, 32      + Kepler support
#               + Unified memory programming
#   35          + Dynamic parallelism support
#   50, 52, 53  + Maxwell support
#   60, 61, 62  + Pascal support
#   70, 72      + Volta support
#   75          + Turing support

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

target_compile_options(geantx-cuda INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)

target_link_libraries(geantx-cuda INTERFACE ${CUDA_LIBRARIES})
target_include_directories(geantx-cuda INTERFACE
    ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


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
set(${PROJECT_NAME}_INCLUDE_DIRECTORIES )

# system include dirs
set(${PROJECT_NAME}_SYSTEM_INCLUDE_DIRECTORIES
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