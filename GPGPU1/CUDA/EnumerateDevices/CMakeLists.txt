cmake_minimum_required(
  VERSION
    3.8 # CUDA language support
)

project(CUDA-EnumerateDevices
  LANGUAGES
    CXX
    CUDA
)

add_executable(${PROJECT_NAME}
  EnumerateDevices.cu
)