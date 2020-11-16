cmake_minimum_required(
  VERSION 3.7
)

project(Cpp-Pi
  LANGUAGES CXX
)

find_package(Threads REQUIRED)

foreach(Ver RANGE 1 2)

set(Sources MC-Pi-v${Ver}.cpp)

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
      Threads::Threads
  )

  source_group("Sources" FILES ${Sources})

endforeach()