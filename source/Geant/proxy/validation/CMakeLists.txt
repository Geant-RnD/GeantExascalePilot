
file(GLOB ValidationHeaders ${CMAKE_CURRENT_LIST_DIR}/*.cuh)
file(GLOB ValidationSources ${CMAKE_CURRENT_LIST_DIR}/*.cu)

add_library(validation INTERFACE)
target_include_directories(validation INTERFACE ${CMAKE_CURRENT_LIST_DIR})

add_geant_google_test(test_process
    SOURCES         test_process.cu
                    ${ValidationSources}
                    ${ValidationtHeaders}
    LINK_LIBRARIES  proxy
                    validation)
