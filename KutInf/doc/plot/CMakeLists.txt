cmake_minimum_required (VERSION 2.8.11)

project(Plots LANGUAGES NONE)

# Find Gnuplot
find_package (Gnuplot REQUIRED)

# Target that generates data file
add_custom_command(COMMAND ${GNUPLOT_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/sin.plt
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   OUTPUT ${CMAKE_BINARY_DIR}/sin.png
                   DEPENDS ${PROJECT_SOURCE_DIR}/script/sin.plt data
                   COMMENT "Generating plot")

add_custom_target(plot DEPENDS ${CMAKE_BINARY_DIR}/sin.png)

