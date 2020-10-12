cmake_minimum_required(
  VERSION
    3.8 # CUDA language support
)

project(CUDA-Reduce
  LANGUAGES
    CXX
    CUDA
)

add_executable(reduce0     reduce0.cu)
add_executable(reduce1     reduce1.cu)
add_executable(reduce2     reduce2.cu)
add_executable(reduce3     reduce3.cu)
add_executable(reduce4     reduce4.cu)
add_executable(reduce5     reduce5.cu)
add_executable(reduce6     reduce6.cu)