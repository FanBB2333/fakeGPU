add_library(fake_gpu_core OBJECT
    global_state.cpp
    device.cpp
)

target_include_directories(fake_gpu_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Needs generated Position Independent Code for shared lib linkage
set_property(TARGET fake_gpu_core PROPERTY POSITION_INDEPENDENT_CODE ON)
