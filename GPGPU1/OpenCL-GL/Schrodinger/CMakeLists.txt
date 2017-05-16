#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.0)

# The name of our project
project (Schrodinger LANGUAGES CXX)

# Adding source code files according to configuration
set (Files_HDRS ${PROJECT_SOURCE_DIR}/include/Schrodinger.hpp)

set (Files_SRCS ${PROJECT_SOURCE_DIR}/source/Schrodinger.cpp)

set (Files_KRNS ${PROJECT_SOURCE_DIR}/kernel/Schrodinger.cl)

set (Files_SHDS ${PROJECT_SOURCE_DIR}/shader/Schrodinger-fragment.glsl
                ${PROJECT_SOURCE_DIR}/shader/Schrodinger-vertex.glsl)

# Generate the configuration file for application to locate kernel files
configure_file ("${PROJECT_SOURCE_DIR}/include/Schrodinger-config.in.hpp"
                "${CMAKE_BINARY_DIR}/include/Schrodinger-config.hpp")

list (APPEND Files_HDRS "${CMAKE_BINARY_DIR}/include/Schrodinger-config.hpp")

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS} ${Files_KRNS} ${Files_SHDS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})
source_group ("Kernels" FILES ${Files_KRNS})
source_group ("Shaders" FILES ${Files_SHDS})

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_BUILD})

# Append our project's include directory to the "#include <>" paths as well as dependencies
target_include_directories (${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/
                                                   ${CMAKE_BINARY_DIR}/include/
                                                   ${OpenCL_INCLUDE_DIRS}
                                                   ${SFML_INCLUDE_DIR})

# Link dependencies
target_link_libraries (${PROJECT_NAME} ${OpenCL_LIBRARIES}
                                       Threads::Threads
                                       GLEW::GLEW
                                       ${SFML_LIBRARIES}
                                       $<$<BOOL:${WIN32}>:glm>)

# Compiler specific options
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes") # cl2.hpp contains attributes on basic types (cl_int, cl_double, etc.)
endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")

# Specify required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 14
                                                  CXX_STANDARD_REQUIRED ON)
