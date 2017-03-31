#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.0)

# The name of our project
project (OpenCL-C-API-demo)

# Set base directory of project
set (Project_Base_Dir ${PROJECT_SOURCE_DIR})

# Append our project's include directory to the "#include <>" paths as well as dependencies
include_directories (${Project_Base_Dir}/inc/)
include_directories (${OpenCL_INCLUDE_DIRS})

# Adding source code files according to configuration
set (Files_HDRS)
set (Files_SRCS)

set (Files_HDRS ${Files_HDRS} ${Project_Base_Dir}/inc/OpenCL-C-API.h)

set (Files_SRCS ${Files_SRCS} ${Project_Base_Dir}/src/OpenCL-C-API.c)
set (Files_SRCS ${Files_SRCS} ${Project_Base_Dir}/src/OpenCL-C-API.cl)

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_BUILD})

# Link dependencies
set (Additional_Libs ${OpenCL_LIBRARIES})
target_link_libraries (${PROJECT_NAME} ${Additional_Libs})
