cmake_minimum_required(VERSION 3.0.0)

project(OpenCL-GL-NBody
  LANGUAGES CXX
)

find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} nbody_gl.cpp)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    OpenCL::OpenCL
    GLEW::GLEW
    glfw
)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CL_TARGET_OPENCL_VERSION=120
)

file(COPY vertex.glsl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY fragment.glsl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY grav.cl DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
