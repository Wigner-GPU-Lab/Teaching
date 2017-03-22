cmake_minimum_required (VERSION 2.8.12)
# CMakeLists files in this project can
# refer to the root source directory of the project as ${RESEARCH_SOURCE_DIR} and
# to the root binary directory of the project as ${RESEARCH_BINARY_DIR}. 
project (CMake_Example)

# Recurse into the "phys" and "app" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
add_subdirectory (phys)
add_subdirectory (app)
add_subdirectory (doc)
