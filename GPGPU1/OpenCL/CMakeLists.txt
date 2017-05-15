# Behavioural options for the OpenGL samples
option (BUILD_CLFFT "Build clFFT library samples" ON)

# Find dependencies
find_package (OpenCL REQUIRED)
if (BUILD_CLFFT)
  find_package (clFFT REQUIRED)
endif (BUILD_CLFFT)

# Recurse into sample code directories
add_subdirectory (OpenCL-C-API-minimal)
add_subdirectory (OpenCL-C-API)
add_subdirectory (OpenCL-C++-API)
add_subdirectory (OpenCL-MatrixMul-C-API)

if (BUILD_CLFFT)
  add_subdirectory (clFFT-1D)
endif (BUILD_CLFFT)