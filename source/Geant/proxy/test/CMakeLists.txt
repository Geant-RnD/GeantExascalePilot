
file(GLOB BasicCpuTransportHeaders ${CMAKE_CURRENT_LIST_DIR}/BasicCpuTransport/*.hpp)
add_library(basic-cpu-transport INTERFACE)
target_include_directories(basic-cpu-transport INTERFACE ${CMAKE_CURRENT_LIST_DIR})

add_geant_google_test(TestBasicCpuTransport
    SOURCES         test_BasicCpuTransport.cpp
                    ${BasicCpuTransportHeaders}
    LINK_LIBRARIES  proxy
                    basic-cpu-transport)
