#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.0)
	
# The name of our solution
project (GPGPU-speci)
	
# Behavioural options for the solution
option (BUILD_OPENCL "Build OpenCL samples" ON)
option (BUILD_SYCL "Build SYCL samples" OFF)
option (BUILD_CUDA "Build CUDA samples" OFF)
option (BUILD_AMP "Build C++AMP samples" OFF)
	
# OpenCL
if (BUILD_OPENCL)
  find_package (OpenCL REQUIRED)
endif (BUILD_OPENCL)

# CUDA
if (BUILD_CUDA)
  find_package (CUDA REQUIRED)
endif (BUILD_CUDA)
	
#
# Set platform-independent variables
#
  # Use solution folders in IDEs
  set_property (GLOBAL PROPERTY USE_FOLDERS ON)

  # Override predefined default output directories
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
	
#
# Set platform-specific variables
#
  # If compiling on Windows
  if (WIN32)
    find_package (Threads REQUIRED)
  endif (WIN32)

  # If compiling under Linux
  if (UNIX)
    find_package (Threads REQUIRED)
  endif (UNIX)

  # If compiling under OSX
  if (APPLE)
    find_package (Threads REQUIRED)
  endif (APPLE)

#
# Set compiler-specific variables
#
  # If compiling with MSVC
  if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
  endif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
  
  # If compiling with MSVC++
  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  endif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  
  # If compiling with gcc
  if (CMAKE_C_COMPILER_ID MATCHES "GNU")
  endif (CMAKE_C_COMPILER_ID MATCHES "GNU")
		
  # If compiling with g++
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
		
#
# Process targets based on user-input
#
  if (BUILD_OPENCL)
    add_subdirectory (OpenCL-C-API-minimal)
    add_subdirectory (OpenCL-C-API)
	add_subdirectory (OpenCL-C++-API)
  endif (BUILD_OPENCL)

  if (BUILD_SYCL)
    add_subdirectory (SYCL)
  endif (BUILD_SYCL)

  if (BUILD_CUDA)
    add_subdirectory (CUDA)
  endif (BUILD_CUDA)

  if (BUILD_AMP)
    add_subdirectory (C++AMP)
  endif (BUILD_AMP)