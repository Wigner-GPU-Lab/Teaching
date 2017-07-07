cmake_minimum_required (VERSION 2.8.11)

project(Doc LANGUAGES NONE)

# Include useful macros
list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include (UseLATEX)

set(SIN "${CMAKE_BINARY_DIR}/sin.png")

# Target that generates documentation
add_latex_document(src/latex_template.tex
                   CONFIGURE src/latex_template.tex
                   TARGET_NAME doc
                   INPUTS src/nyilatkozat.tex src/abeld.bst
                   BIBFILES src/references.bib
                   IMAGE_DIRS img
                   DEPENDS src/latex_template.tex plot
                   EXCLUDE_FROM_ALL)

# Process plot CMake script
add_subdirectory(plot)

