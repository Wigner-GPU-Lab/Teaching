cmake_minimum_required(VERSION 3.0.0)
project(CTest_Test VERSION 0.1.0)

include(CTest)
enable_testing()

if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif (MSVC)

add_executable(CTest_Test main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17
                                                 CXX_STANDARD_REQUIRED ON
                                                 CXX_EXTENSIONS OFF)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                               $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)

if(BUILD_TESTING)
  add_executable(test1 test1.cpp)
  add_test(NAME test1 COMMAND test1)
  
  add_executable(test2 test2.cpp)
  add_test(NAME test2 COMMAND test2)
endif(BUILD_TESTING)
