# Find dependencies
find_package (OpenCL REQUIRED)

# Recurse into sample code directories
add_subdirectory (OpenCL-C-API-minimal)
add_subdirectory (OpenCL-C-API)
add_subdirectory (OpenCL-C++-API)
add_subdirectory (OpenCL-MatrixMul-C-API)