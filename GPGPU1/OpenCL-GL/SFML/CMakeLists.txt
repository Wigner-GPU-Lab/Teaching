#
# Find dependencies
#
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(SFML COMPONENTS system window graphics REQUIRED)
# Windows systems require a CMake-aware glm installation
# Unix-like systems require a package managed installation
if (WIN32)
  find_package(glm)
endif (WIN32)

# Recurse into sample code directories
add_subdirectory (Schrodinger)
