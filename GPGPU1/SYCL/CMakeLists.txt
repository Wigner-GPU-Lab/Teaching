# Append to module path for locating FindComputeCpp.cmake
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

# Find dependencies
find_package (ComputeCpp REQUIRED)

# Recurse into sample code directories
add_subdirectory (SYCL-MatrixMul)
