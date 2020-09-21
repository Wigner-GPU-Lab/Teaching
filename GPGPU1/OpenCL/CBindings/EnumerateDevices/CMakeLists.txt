cmake_minimum_required(
  VERSION 3.7
)

project(OpenCL-C-EnumerateDevices
  LANGUAGES C
)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

set(Sources EnumerateDevices.c)

add_executable(${PROJECT_NAME}
  ${Sources}
)

target_compile_features(${PROJECT_NAME}
  PRIVATE
    c_std_11
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    C_EXTENSIONS OFF
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    OpenCL::OpenCL
    Threads::Threads
)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CL_TARGET_OPENCL_VERSION=120
)

source_group("Sources" FILES ${Sources})