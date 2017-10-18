#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.8)

# The name of our project
project (CUDA-C++-API LANGUAGES CXX CUDA)

# Adding source code files according to configuration
set (Files_HDRS ${PROJECT_SOURCE_DIR}/include/CUDA-C++-API.hpp)
set (Files_SRCS ${PROJECT_SOURCE_DIR}/source/CUDA-C++-API.cu)

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_BUILD})

# Append our project's include directory to the "#include <>" paths as well as dependencies
target_include_directories (${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include/)

# Link dependencies
target_link_libraries (${PROJECT_NAME} PRIVATE Threads::Threads)

# Specify required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CUDA_STANDARD 14
                                                  CUDA_STANDARD_REQUIRED ON)

# In CUDA 9.0, this flag enables 'auto lambda = [] __device__ (){};' syntax
set_source_files_properties(${Files_SRCS} PROPERTIES COMPILE_FLAGS --expt-extended-lambda)