cmake_minimum_required(VERSION 3.2.2)

project(SYCL-WithSFML LANGUAGES CXX)

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCL REQUIRED)
find_package(SFML 2 COMPONENTS graphics REQUIRED)
find_package(glm REQUIRED)
find_package(Threads)

add_executable(${PROJECT_NAME} Main.cpp)

set_target_properties(${PROJECT_NAME}
                      PROPERTIES CXX_STANDARD 17
                                 CXX_STANDARD_REQUIRED ON)

target_link_libraries(${PROJECT_NAME} PRIVATE sfml-graphics
                                              OpenGL::GL
                                              GLEW::GLEW
                                              OpenCL::OpenCL
                                              Threads::Threads
                                              glm)

add_sycl_to_target(TARGET ${PROJECT_NAME}
                   SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp)