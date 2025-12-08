add_library(fake_gpu_monitor OBJECT
    monitor.cpp
)

target_include_directories(fake_gpu_monitor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fake_gpu_monitor PRIVATE ../core)
set_property(TARGET fake_gpu_monitor PROPERTY POSITION_INDEPENDENT_CODE ON)
