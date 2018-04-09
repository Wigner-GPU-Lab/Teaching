cmake_minimum_required(VERSION 3.5)

project(CppSampleCodes LANGUAGES CXX)

set(Samples file_io)

foreach(Sample ${Samples})
  add_executable(${Sample} "${Sample}.cpp")

  set_target_properties(${Sample} PROPERTIES CXX_STANDARD 14
                                             CXX_STANDARD_REQUIRED ON
											 CXX_EXTENSIONS OFF)
endforeach(Sample)