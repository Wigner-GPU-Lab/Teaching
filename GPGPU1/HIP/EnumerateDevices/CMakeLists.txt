cmake_minimum_required(
  VERSION 3.8 # cxx_std_17
)

project(HIP-EnumerateDevices
  LANGUAGES CXX
)

option(USE_HIP_CPU "Prefer HIP-CPU runtime instead of HW acceleration" OFF)

add_executable(${PROJECT_NAME}
  EnumerateDevices.hip.cpp
)

if(USE_HIP_CPU)
  find_package(hip_cpu_rt REQUIRED)

  target_link_libraries(${PROJECT_NAME}
    PUBLIC
      hip_cpu_rt::hip_cpu_rt
  )

  target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_17
  )
else()
  set(AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908" CACHE STRING "List of specific machine types for library to target")
  find_package(hip REQUIRED CONFIG)

  target_link_libraries(${PROJECT_NAME}
    hip::device
  )
endif()