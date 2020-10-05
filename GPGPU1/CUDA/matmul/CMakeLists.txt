cmake_minimum_required(
  VERSION
    3.8 # CUDA language support
)

project(CUDA-MeasureKernelExecution
  LANGUAGES
    CXX
    CUDA
)

add_executable(matmul_naive matmul_naive.cu)
add_executable(matmul_improved1 matmul_improved1.cu)
add_executable(matmul_improved2 matmul_improved2.cu)
add_executable(matmul_improved3 matmul_improved3.cu)