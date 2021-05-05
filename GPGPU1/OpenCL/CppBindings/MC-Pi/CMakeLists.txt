cmake_minimum_required(
  VERSION 3.7
)

project(OpenCL-Cpp-MC-Pi
  LANGUAGES CXX
)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)
find_package(clRNG REQUIRED)

set(Sources MC-Pi.cpp)
set(Kernels MC-Pi.cl)

add_executable(${PROJECT_NAME}
  ${Sources}
  ${Kernels}
)

target_compile_features(${PROJECT_NAME}
  PRIVATE
    cxx_std_17
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CXX_EXTENSIONS OFF
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    OpenCL::OpenCL
    Threads::Threads
    clRNG
)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_ENABLE_EXCEPTIONS
)

source_group("Sources" FILES ${Sources})
source_group("Kernels" FILES ${Kernels})