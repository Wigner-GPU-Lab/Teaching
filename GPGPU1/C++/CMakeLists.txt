# Behavioural options for the OpenGL samples
option (BUILD_NBODY "Build NBody samples" ON)
option (BUILD_STL "Build STL samples" ON)

# Recurse into sample code directories
if (BUILD_NBODY)
  add_subdirectory (NBody/cmake)
endif (BUILD_NBODY)