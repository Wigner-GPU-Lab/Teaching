#
# Find dependencies
#
find_package (OpenCL REQUIRED)

# Recurse into sample dirs based on user-input
if (USE_SFML)
  add_subdirectory (SFML)
endif (USE_SFML)

if (USE_Qt)
  add_subdirectory (Qt)
endif (USE_Qt)