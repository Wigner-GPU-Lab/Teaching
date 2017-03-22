cmake_minimum_required (VERSION 2.8.11)

project(Doc LANGUAGES NONE)

# Include useful macros
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include (UseLATEX)

# Target that generates documentation
add_latex_document(src/latex_template.tex
                   TARGET_NAME doc
                   INPUTS src/nyilatkozat.tex
                   BIBFILES src/references.bib
                   #IMAGES img/elte.eps ${CMAKE_BINARY_DIR}/sin.eps
                   DEPENDS src/latex_template.tex plot
                   EXCLUDE_FROM_ALL)

# Process plot CMake script
add_subdirectory(plot)
