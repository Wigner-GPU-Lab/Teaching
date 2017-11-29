Science Targeted Programming of Graphical Processors I.
===================


This repository contains sample codes used throughout the slides of the course. All samples are meant to be cross-platform (in terms of operating systems) and cross-vendor (in terms of GPU vendors). Should you have troubles building any of the samples in a given scenario, we accept pull requests to make both the code and build scripts more robust.

Building the samples
-------------

Build automation is done via CMake and should provide an out-of-the-box experience on most systems. Samples rely on ISO C++ conforming compilers of various versions.

## Requirements
All samples require:
* CMake version 3.0+
* ISO C++11/14 compiler

OpenCL samples require:
* OpenCL 1.2+ development files
* OpenCL 1.2+ implementation
* clFFT 2.12+

SYCL samples require:
* ComputeCpp SDK
* _(triSYCL support inbound)_

CUDA samples require:
* CUDA SDK 9.0+

OpenGL samples require:
* OpenGL 3.3+ implementation
* SFML 2.0+
* GLEW 2.0+
* GLM 0.9.0+
* Qt 5.7+

The build scripts by default build all the samples. One can selectively opt out of building specific sample suites via setting the following CMake options to `OFF`:

* `BUILD_OPENCL` to disable all OpenCL samples
    * `USE_CLFFT` to disable only the clFFT samples
* `BUILD_SYCL` to disable all the SYCL samples
* `BUILD_OPENGL` to disable all OpenGL samples
    * `USE_SFML` to disable only the SFML based OpenGL samples
    * `USE_QT` to disable only the Qt based OpenGL samples

Leaving any of the compute APIs (OpenCL/SYCL/CUDA) and OpenGL enabled will result in building interop samples when available, using the remaining active windowing libraries.

_Should you be confused on how to use CMake, please consult our CMake tutorial [here](../CMake)._

Platform-specific build guidelines can be found in the docs:
* [Windows](./docs/Windows.md)
* [Ubuntu 16.04](./docs/Ubuntu.md)