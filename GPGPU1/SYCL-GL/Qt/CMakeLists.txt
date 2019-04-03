#
# Find dependencies
#
find_package (Qt5 5.5 CONFIG REQUIRED Core Gui OpenGL)
find_package (OpenGL REQUIRED)

# Recurse into sample code directories
add_subdirectory (NBody)
add_subdirectory (Conway)
