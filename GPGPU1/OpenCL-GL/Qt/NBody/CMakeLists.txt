# Scripts want to rely on stock FindOpenCL.cmake
cmake_minimum_required (VERSION 3.1)

set (Target_Name NBody)

# Always include current directory for all compile commands 
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find Qt5 installation
#find_package (Qt5 5.7 CONFIG REQUIRED Core Gui OpenGL)

# Specify source and header files
set (Sources source/Main.cpp
             source/NBody.cpp
             source/InteropWindow.cpp)

set (Headers include/InteropWindow.hpp
             include/NBody.hpp)

set (Shaders shader/Vertex.glsl
             shader/Fragment.glsl)

set (Kernels kernel/Kernels.cl)

# Declare executable target
add_executable(${Target_Name} ${Sources} ${Headers} ${Shaders} ${Kernels})

# Target include directories
target_include_directories (${Target_Name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Target link directives
target_link_libraries(${Target_Name} PUBLIC Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::WinMain
                                            OpenCL::OpenCL)

# Automatically invoke Qt tools on source files
set_target_properties(${Target_Name} PROPERTIES CXX_STANDARD 11
                                                CXX_STANDARD_REQUIRED ON
                                                AUTOMOC ON)

# Create source groups for IDE
source_group ("Sources" FILES ${Sources})
source_group ("Headers" FILES ${Headers})
source_group ("Shaders" FILES ${Shaders})
source_group ("Kernels" FILES ${Kernels})
