#
# Find dependencies
#
# Append to module path for locating FindComputeCpp.cmake
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

# Find dependencies
find_package (ComputeCpp REQUIRED)

# Recurse into sample dirs based on user-input
if (USE_Qt)
  add_subdirectory (Qt)
endif (USE_Qt)