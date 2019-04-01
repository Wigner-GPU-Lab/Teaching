# Scripts want to rely on stock FindOpenCL.cmake
cmake_minimum_required (VERSION 3.1)

set (Target_Name NBody-SYCL)

# Always include current directory for all compile commands 
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Generate the configuration file for application to locate shader files
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/NBody-Config.in.hpp"
                "${CMAKE_CURRENT_BINARY_DIR}/include/NBody-Config.hpp")

# Specify source and header files
set (Sources source/Main.cpp
             source/NBody.cpp
             source/InteropWindow.cpp)

set (Headers include/InteropWindow.hpp
             include/NBody.hpp
             ${CMAKE_CURRENT_BINARY_DIR}/include/NBody-Config.hpp)

set (Shaders shader/Vertex.glsl
             shader/Fragment.glsl)

# Declare executable target
add_executable(${Target_Name} ${Sources} ${Headers} ${Shaders})

# Target include directories
target_include_directories (${Target_Name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                  ${CMAKE_CURRENT_BINARY_DIR}/include
                                                  ${Qt5Gui_PRIVATE_INCLUDE_DIRS}
                                                  ${ComputeCpp_INCLUDE_DIRS})

# Target link directives
target_link_libraries(${Target_Name} PUBLIC Qt5::Core Qt5::Gui Qt5::OpenGL $<$<PLATFORM_ID:Windows>:Qt5::WinMain>
                                            OpenGL::GL)

# Automatically invoke Qt tools on source files
set_target_properties(${Target_Name} PROPERTIES CXX_STANDARD 14
                                                CXX_STANDARD_REQUIRED ON
                                                AUTOMOC ON)

# Apply the ComputeCpp compiler to select source files (containing SYCL code)
add_sycl_to_target(TARGET ${Target_Name}
                   SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/NBody.cpp)

# Create source groups for IDE
source_group ("Sources" FILES ${Sources})
source_group ("Headers" FILES ${Headers})
source_group ("Shaders" FILES ${Shaders})
