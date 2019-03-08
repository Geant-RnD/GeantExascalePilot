################################################################################
#
#        Handles the build options
#
################################################################################

include(GNUInstallDirs)
include(Compilers)

# ---------------------------------------------------------------------------- #
# set the compiler flags
add_c_flag_if_avail("-W")
add_c_flag_if_avail("-Wall")
add_c_flag_if_avail("-Wextra")
add_c_flag_if_avail("-Wno-attributes")
add_c_flag_if_avail("-Wno-unused-variable")
add_c_flag_if_avail("-Wno-unknown-pragmas")
add_c_flag_if_avail("-Wno-unused-parameter")
add_c_flag_if_avail("-Wunused-but-set-parameter")

# SIMD OpenMP
add_c_flag_if_avail("-fopenmp-simd")
# Intel floating-point model
add_c_flag_if_avail("-fp-model=precise")

# OpenMP (non-SIMD)
if(GEANT_USE_OPENMP)
    add_c_flag_if_avail("-mp=nonuma")
    add_cxx_flag_if_avail("-mp=nonuma")
endif(GEANT_USE_OPENMP)

add_cxx_flag_if_avail("-W")
add_cxx_flag_if_avail("-Wall")
add_cxx_flag_if_avail("-Wextra")
add_cxx_flag_if_avail("-Wno-attributes")
add_cxx_flag_if_avail("-Wno-unused-value")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-unused-parameter")
add_cxx_flag_if_avail("-Wunused-but-set-parameter")
add_cxx_flag_if_avail("-faligned-new")

# SIMD OpenMP
add_cxx_flag_if_avail("-fopenmp-simd")
# Intel floating-point model
add_cxx_flag_if_avail("-fp-model=precise")

if(GEANT_USE_ARCH)
    if(CMAKE_C_COMPILER_IS_INTEL)
        add_c_flag_if_avail("-xHOST")
        if(GEANT_USE_AVX512)
            add_c_flag_if_avail("-axMIC-AVX512")
        endif()
    else()
        add_c_flag_if_avail("-march=native")
        add_c_flag_if_avail("-mtune=native")
        add_c_flag_if_avail("-msse2")
        add_c_flag_if_avail("-msse3")
        add_c_flag_if_avail("-mssse3")
        add_c_flag_if_avail("-msse4")
        add_c_flag_if_avail("-msse4.1")
        add_c_flag_if_avail("-msse4.2")
        add_c_flag_if_avail("-mavx")
        add_c_flag_if_avail("-mavx2")
        if(GEANT_USE_AVX512)
            add_c_flag_if_avail("-mavx512f")
            add_c_flag_if_avail("-mavx512pf")
            add_c_flag_if_avail("-mavx512er")
            add_c_flag_if_avail("-mavx512cd")
            add_c_flag_if_avail("-mavx512vl")
            add_c_flag_if_avail("-mavx512bw")
            add_c_flag_if_avail("-mavx512dq")
            add_c_flag_if_avail("-mavx512ifma")
            add_c_flag_if_avail("-mavx512vbmi")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_IS_INTEL)
        add_cxx_flag_if_avail("-xHOST")
        if(GEANT_USE_AVX512)
            add_cxx_flag_if_avail("-axMIC-AVX512")
        endif()
    else()
        add_cxx_flag_if_avail("-march=native")
        add_cxx_flag_if_avail("-mtune=native")
        add_cxx_flag_if_avail("-msse2")
        add_cxx_flag_if_avail("-msse3")
        add_cxx_flag_if_avail("-mssse3")
        add_cxx_flag_if_avail("-msse4")
        add_cxx_flag_if_avail("-msse4.1")
        add_cxx_flag_if_avail("-msse4.2")
        add_cxx_flag_if_avail("-mavx")
        add_cxx_flag_if_avail("-mavx2")
        if(GEANT_USE_AVX512)
            add_cxx_flag_if_avail("-mavx512f")
            add_cxx_flag_if_avail("-mavx512pf")
            add_cxx_flag_if_avail("-mavx512er")
            add_cxx_flag_if_avail("-mavx512cd")
            add_cxx_flag_if_avail("-mavx512vl")
            add_cxx_flag_if_avail("-mavx512bw")
            add_cxx_flag_if_avail("-mavx512dq")
            add_cxx_flag_if_avail("-mavx512ifma")
            add_cxx_flag_if_avail("-mavx512vbmi")
        endif()
    endif()
endif()


if(GEANT_USE_SANITIZER)
    add_c_flag_if_avail("-fsanitize=${SANITIZER_TYPE}")
    add_cxx_flag_if_avail("-fsanitize=${SANITIZER_TYPE}")
endif()

if(GEANT_USE_COVERAGE)
    add_c_flag_if_avail("-ftest-coverage")
    if(c_ftest_coverage)
        add(${PROJECT_NAME}_C_FLAGS "-fprofile-arcs")
    endif()
    add_cxx_flag_if_avail("-ftest-coverage")
    if(cxx_ftest_coverage)
        add(${PROJECT_NAME}_CXX_FLAGS "-fprofile-arcs")
        add(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs")
        add_feature(CMAKE_EXE_LINKER_FLAGS "Linker flags")
    endif()
endif()

# ---------------------------------------------------------------------------- #
# user customization
to_list(_CFLAGS "${CFLAGS};$ENV{CFLAGS}")
foreach(_FLAG ${_CFLAGS})
    add_c_flag_if_avail("${_FLAG}")
endforeach()

to_list(_CXXFLAGS "${CXXFLAGS};$ENV{CXXFLAGS}")
foreach(_FLAG ${_CXXFLAGS})
    add_cxx_flag_if_avail("${_FLAG}")
endforeach()

