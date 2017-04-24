project(computecpptest)

cmake_minimum_required(VERSION 3.2.2)

set(COMPUTECPP_PACKAGE_ROOT_DIR /opt/Codeplay/ComputeCpp/latest)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
include(FindOpenCL)
include(FindComputeCpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set include ComputeCpp directories
include_directories(SYSTEM ${COMPUTECPP_INCLUDE_DIRECTORY})
include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})
include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

add_executable(matmul.out ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp)
target_compile_options(matmul.out PUBLIC -std=c++11 -Wall)
add_sycl_to_target(matmul.out ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR})
