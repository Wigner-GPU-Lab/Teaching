# Behavioural options for the OpenGL samples
option (USE_SFML "Use SFML windowing" ON)

# Recurse into sample dirs based on user-input
if (USE_SFML)
  add_subdirectory (SFML)
endif (USE_SFML)
