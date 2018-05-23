#########################
#                       #
#         CMake         #
#                       #
#########################

# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 2.8)

# The name of our project
project (manybody-c++-v5-parallel)

# Set base directory of project
set (Project_Base_Dir ${PROJECT_SOURCE_DIR}/../../)

# Adding source code files according to configuration
set (Files_HDRS)
set (Files_SRCS)

set (Files_HDRS ${Files_SRCS} ${Project_Base_Dir}/inc/particle.hpp)

set (Files_SRCS ${Files_SRCS} ${Project_Base_Dir}/src/${PROJECT_NAME}.cpp)

# Variable to hold ALL files to build and be visible in IDE
set (Files_BUILD ${Files_HDRS} ${Files_SRCS})

# Create filters for IDEs
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_BUILD})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${PROJECT_NAME} PUBLIC ${Project_Base_Dir}/inc/)

# Link dependencies
target_link_libraries (${PROJECT_NAME} Threads::Threads)

# Create project groups for IDEs
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 14
                                                  CXX_STANDARD_REQUIRED ON
                                                  CXX_EXTENSIONS OFF
                                                  FOLDER "C++\\NBody")

#########################
#                       #
#         CTest         #
#                       #
#########################

# Add unit test
if (CP_BUILD_TESTS)
		
  add_test(NAME ${PROJECT_NAME}
           COMMAND ${PROJECT_NAME} --input ${CP_MANYBODY_TEST_DATA} --validate ${CP_MANYBODY_TEST_VALIDATE})
		
endif (CP_BUILD_TESTS)