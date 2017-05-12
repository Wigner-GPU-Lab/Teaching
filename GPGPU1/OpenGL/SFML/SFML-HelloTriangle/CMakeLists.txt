#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.2.2)

# The name of our project
project (SFML_HelloTriangle)

# Adding source code files according to configuration
set (Files_HDRS ${PROJECT_SOURCE_DIR}/include/SFML-HelloTriangle.hpp)

set (Files_SRCS ${PROJECT_SOURCE_DIR}/source/SFML-HelloTriangle.cpp)

set (Files_SHDS ${PROJECT_SOURCE_DIR}/shader/SFML-HelloTriangle-fragment.glsl
                ${PROJECT_SOURCE_DIR}/shader/SFML-HelloTriangle-vertex.glsl)

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS} ${Files_SHDS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})
source_group ("Shaders" FILES ${Files_SHDS})

# Generate the configuration file for application to locate kernel files
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/SFML-HelloTriangle-config.in.hpp"
                "${CMAKE_BINARY_DIR}/include/SFML-HelloTriangle-config.hpp")

# Specify executable sources
add_executable(${PROJECT_NAME} ${Files_BUILD})

# Append our project's include directory to the "#include <>" paths as well as dependencies
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                  ${CMAKE_BINARY_DIR}/include
                                                  ${SFML_INCLUDE_DIR})
# Link dependencies
target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES}
                                             GLEW::GLEW
											 ${SFML_LIBRARIES}
                                             $<$<BOOL:${WIN32}>:glm>)
# Specify required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 11
                                                  CXX_STANDARD_REQUIRED ON)
