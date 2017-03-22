cmake_minimum_required (VERSION 2.8.11)

project(Application LANGUAGES CXX)

# Add executable called "Application" that is built from the source files
# 'main.cpp'. The extensions are automatically found.
add_executable (${PROJECT_NAME} src/main.cpp)

# Make sure the compiler can find include files for our Application sources
target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Link the executable to the Phys library. Since the Phys library has
# public include directories we will use those link directories when building
# Application
target_link_libraries (${PROJECT_NAME} LINK_PUBLIC Phys)

# Enable C++11 language support
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

# Target that generates data file
add_custom_command(COMMAND ${PROJECT_NAME}
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   OUTPUT ${CMAKE_BINARY_DIR}/sin.dat
                   BYPRODUCTS ${CMAKE_BINARY_DIR}/sin.dat
                   DEPENDS ${PROJECT_NAME}
                   COMMENT "Generating data set")

add_custom_target(data DEPENDS ${CMAKE_BINARY_DIR}/sin.dat)
