cmake_minimum_required(VERSION 3.16.0)

project(CUDA-GL-NBody
  LANGUAGES CUDA
)

find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} nbody_gl.cu)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    GLEW::GLEW
    glfw
)

file(COPY vertex.glsl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY fragment.glsl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
