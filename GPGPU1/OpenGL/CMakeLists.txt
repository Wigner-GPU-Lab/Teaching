# Behavioural options for the OpenGL samples
option (USE_SFML "Use SFML windowing" ON)
option (USE_Qt "Use Qt windowing" ON)

# Recurse into sample dirs based on user-input
if (USE_SFML)
  add_subdirectory (SFML)
endif (USE_SFML)

if (USE_Qt)
  add_subdirectory (Qt)
endif (USE_Qt)
