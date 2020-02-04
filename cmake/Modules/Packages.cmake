################################################################################
#
#                               Handle the external packages
#
################################################################################

include(FindPackageHandleStandardArgs)

add_library(geantx-external INTERFACE)

add_library(geantx-cuda INTERFACE)
add_library(geantx-threading INTERFACE)
add_library(geantx-coverage INTERFACE)

set(GEANTX_EXTERNAL_INTERFACES
    geantx-cuda
    geantx-threading
)

target_link_libraries(geantx-external INTERFACE ${GEANTX_EXTERNAL_INTERFACES})

# if option is enabled, have geantx-external always provide tools listed below:
if(GEANT_USE_COVERAGE)
    target_link_libraries(geantx-external INTERFACE geantx-coverage)
endif()


################################################################################
#
#                               CUDA
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

target_compile_options(geantx-cuda INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)

target_link_libraries(geantx-cuda INTERFACE ${CUDA_LIBRARIES})
target_include_directories(geantx-cuda INTERFACE
    ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


################################################################################
#
#                               Threading
#
################################################################################

if(NOT WIN32)
    set(CMAKE_THREAD_PREFER_PTHREAD ON)
    set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "Use -pthread vs. -lpthread" FORCE)
endif()

find_library(PTHREADS_LIBRARY pthread)
find_package(Threads QUIET)

if(Threads_FOUND)
    target_link_libraries(geantx-threading INTERFACE Threads::Threads)
endif()

if(PTHREADS_LIBRARY AND NOT WIN32)
    target_link_libraries(geantx-threading INTERFACE ${PTHREADS_LIBRARY})
endif()


################################################################################
#
#                               Coverage
#
################################################################################

find_library(GCOV_LIBRARY gcov)

if(GCOV_LIBRARY)
    target_link_libraries(geantx-coverage INTERFACE ${GCOV_LIBRARIES})
else()
    target_link_libraries(geantx-coverage INTERFACE gcov)
endif()

