cmake_minimum_required(
  VERSION 3.7
)

project(OpenCL-Cpp-NBody
  LANGUAGES CXX
)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

foreach(Ver RANGE 1 5)

  set(Sources NBody-v${Ver}.cpp)

  add_executable(${PROJECT_NAME}-v${Ver}
    ${Sources}
  )

  target_compile_features(${PROJECT_NAME}-v${Ver}
    PRIVATE
      cxx_std_17
  )

  set_target_properties(${PROJECT_NAME}-v${Ver}
    PROPERTIES
      CXX_EXTENSIONS OFF
  )

  target_link_libraries(${PROJECT_NAME}-v${Ver}
    PRIVATE
      OpenCL::OpenCL
      Threads::Threads
  )

  target_compile_definitions(${PROJECT_NAME}-v${Ver}
    PRIVATE
      CL_HPP_MINIMUM_OPENCL_VERSION=120
      CL_HPP_TARGET_OPENCL_VERSION=120
      CL_HPP_ENABLE_EXCEPTIONS
  )

  source_group("Sources" FILES ${Sources})

endforeach()