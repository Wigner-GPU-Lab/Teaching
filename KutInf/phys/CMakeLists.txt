cmake_minimum_required (VERSION 2.8.11)

project(Phys LANGUAGES CXX)

# Create a library called "Phys" which includes the source files 'stuff.cpp' and 'more.cpp'.
# The extension is already found. Any number of sources could be listed here. 
add_library (${PROJECT_NAME} inc/phys.hpp src/phys.cpp)

# Make sure the compiler can find include files for our Phys library
# when other libraries or executables link to Phys
target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Enable C++11 language support
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
