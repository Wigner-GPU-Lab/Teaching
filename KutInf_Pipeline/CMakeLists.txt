cmake_minimum_required (VERSION 2.8.12)
# CMakeLists files in this project can
# refer to the root source directory of the project as ${RESEARCH_SOURCE_DIR} and
# to the root binary directory of the project as ${RESEARCH_BINARY_DIR}. 
project (CMakePipeline LANGUAGES CXX)

# Add executable called "CMakePipeline" that is built from the source files
# 'main.cpp'. The extensions are automatically found.
add_executable (CMakePipeline main.cpp)

# Target that generates data file
add_custom_command(COMMAND CMakePipeline
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   OUTPUT ${CMAKE_BINARY_DIR}/sin.dat
                   DEPENDS CMakePipeline
                   COMMENT "Generating data set")

add_custom_target(data ALL DEPENDS ${CMAKE_BINARY_DIR}/sin.dat)

find_package (Gnuplot REQUIRED)

# Target that generates data file
add_custom_command(COMMAND ${GNUPLOT_EXECUTABLE} ${PROJECT_SOURCE_DIR}/sin.plt
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   OUTPUT ${CMAKE_BINARY_DIR}/sin.png
                   DEPENDS ${PROJECT_SOURCE_DIR}/sin.plt data
                   COMMENT "Generating plot")

add_custom_target(plot ALL DEPENDS ${CMAKE_BINARY_DIR}/sin.png)