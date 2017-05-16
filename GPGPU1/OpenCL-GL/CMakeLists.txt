# Behavioural options for the interop samples
# All options have been declared by pure compute and graphics samples

#
# Find dependencies
#
find_package (OpenCL REQUIRED)
if (USE_SFML)
  find_package(OpenGL REQUIRED)
  find_package(GLEW REQUIRED)
  find_package(SFML COMPONENTS system window graphics REQUIRED)
  # Windows systems require a CMake-aware glm installation
  # Unix-like systems require a package managed installation
  if (WIN32)
    find_package(glm)
  endif (WIN32)
endif (USE_SFML)

if (USE_SFML)
  add_subdirectory (Schrodinger)
endif (USE_SFML)