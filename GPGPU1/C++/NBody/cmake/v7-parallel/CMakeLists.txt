#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 2.8)

# The name of our project
project (manybody-c++-v6-parallel)

# Set base directory of project
set (Project_Base_Dir ${PROJECT_SOURCE_DIR}/../../)

# Append our project's CMake directory to CMake's own module path
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Project_Base_Dir}/cmake/Modules/")

# Override predefined default output directories
set (CMAKE_USE_RELATIVE_PATH "true")

# Append our project's include directory to the "#include <>" paths
include_directories (${Project_Base_Dir}/inc/)

# Adding source code files according to configuration
set (Files_HDRS)
set (Files_SRCS)

set (Files_HDRS ${Files_SRCS} ${Project_Base_Dir}/inc/particle.hpp)
set (Files_HDRS ${Files_SRCS} ${Project_Base_Dir}/inc/octree.hpp)

set (Files_SRCS ${Files_SRCS} ${Project_Base_Dir}/src/manybody-c++-v6-parallel.cpp)

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_BUILD})

# Create project groups for IDEs
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER "C++\\NBody")