#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.2.2)

# The name of our project
project (SYCL-MatrixMul)

# Adding source code files according to configuration
set (Files_HDRS)
set (Files_SRCS)

set (Files_HDRS ${Files_HDRS} ${PROJECT_SOURCE_DIR}/include/SYCL-MatrixMul.hpp)

set (Files_SRCS ${Files_SRCS} ${PROJECT_SOURCE_DIR}/source/SYCL-MatrixMul.cpp)

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_BUILD})

# Append our project's include directory to the "#include <>" paths as well as dependencies
target_include_directories (${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                   ${COMPUTECPP_INCLUDE_DIRECTORY})
#target_include_directories (${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})

# Specify required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 11
                                                  CXX_STANDARD_REQUIRED ON)

# Invoke the ComputeCpp compiler and append output to the list of object files for linkage
add_sycl_to_target (${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR} ${Files_SRCS})

##########################################################

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#
## Set include ComputeCpp directories
#include_directories(SYSTEM ${COMPUTECPP_INCLUDE_DIRECTORY})
#include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})
#include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})
#
#add_executable(matmul.out ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp)
#target_compile_options(matmul.out PUBLIC -std=c++11 -Wall)
#add_sycl_to_target(matmul.out ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp 
#                    ${CMAKE_CURRENT_BINARY_DIR})
