#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.0)

# The name of our project
project (OpenCL-C++-API-demo)

# Adding source code files according to configuration
set (Files_HDRS)
set (Files_SRCS)
set (Files_KRNS)

set (Files_HDRS ${Files_HDRS} ${PROJECT_SOURCE_DIR}/include/OpenCL-C++-API.hpp)

set (Files_SRCS ${Files_SRCS} ${PROJECT_SOURCE_DIR}/source/OpenCL-C++-API.cpp)

set (Files_KRNS ${Files_KRNS} ${PROJECT_SOURCE_DIR}/kernel/saxpy.cl)

# Generate the configuration file for application to locate kernel files
configure_file ("${PROJECT_SOURCE_DIR}/include/OpenCL-C++-API-config.in.hpp"
                "${CMAKE_BINARY_DIR}/include/OpenCL-C++-API-config.hpp")

set (Files_HDRS ${Files_HDRS} "${CMAKE_BINARY_DIR}/include/OpenCL-C++-API-config.hpp")

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS} ${Files_KRNS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})
source_group ("Kernels" FILES ${Files_KRNS})

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_BUILD})

# Append our project's include directory to the "#include <>" paths as well as dependencies
target_include_directories (${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories (${PROJECT_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include/)
target_include_directories (${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})

# Link dependencies
target_link_libraries (${PROJECT_NAME} ${OpenCL_LIBRARIES} Threads::Threads)

# Compiler specific options
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # The implementation of cl::copy does not comply with MSVC checked iterators
  target_compile_definitions(${PROJECT_NAME} PRIVATE "-D_SCL_SECURE_NO_WARNINGS")
endif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes") # cl2.hpp contains attributes on basic types (cl_int, cl_double, etc.)
endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")

# Specify required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 14
                                                  CXX_STANDARD_REQUIRED ON)
