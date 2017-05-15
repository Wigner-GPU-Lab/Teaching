Science Targeted Programming of Graphical Processors I.
===================


This repository contains sample codes used throughout the slides of the course. All samples are meant to be cross-platform (in terms of operating systems) and cross-vendor (in terms of GPU vendors). Should you have troubles building any of the samples in a given scenario, we accept pull requests to make both the code and build scripts more robust.

----------

Building the samples
-------------

Build automation is done via CMake and should provide an out-of-the-box experience on most systems. Samples rely on ISO C++ conforming compilers of various versions.

## Requirements
All samples require:
* CMake version 3.0+
* ISO C++11/14 compiler

OpenCL samples require:
* OpenCL 1.2+ implementation

OpenGL samples require:
* OpenGL 3.3+ implementation
* SFML 2.0+
* GLEW 2.0+
* GLM 0.9.0+
* Qt 5.7+

The build scripts by default build all the samples. One can selectively opt out of building specific sample suites via setting the following variables to `OFF`:
* `BUILD_OPENCL` to disable all OpenCL samples
* `BUILD_OPENGL` to disable all OpenGL samples
    * `BUILD_SFML` to disable only the SFML based OpenGL samples
    * `BUILD_QT` to disable only the Qt based OpenGL samples

Leaving both OpenCL and OpenGL enabled will result in building interop samples using the remaining active windowing libraries.

### Windows
We __highly recommend both installing and building 64-bit binaries__ on systems that support it. 32-bit builds on 64-bit enabled operating systems is considered cross-compilation and should be avoided if possible. Search for _x86_64_, _x64_ or _amd64_ markers, all referring to the same architecture, most often referred to as _64-bit_.
#### Installation
* Grab the latest "next-next-finish" CMake installer from the Kitware [downloads section](https://cmake.org/download/).
    * For conveniant usage, either let the installer add cmake.exe to your PATH, or add itt manually.
* We highly recommend building native executables using the Microsoft Visual C++ compiler. There are two recommended ways to install it:
    * When used with the Visual Studio IDE, install Visual Studio 2017 Community Edition (for OSS and small team development) from [here](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=Community&rel=15). Because of the still evolving tooling around CMake Server Mode, it is recommended to always install and update to the latest Visual Studio version available.
        * Bleeding edge, preview versions can be obtained from [here](https://www.visualstudio.com/vs/preview/). Allows for side-by-side installation with production versions.
        * Visual Studio ships with a fork of CMake that drives the IDE experience, though having a more recent seperate install can come in handy.
    * When used from the command-line or through another IDE, one can obtain the __Build Tools for Visual Studio 2017__ from [here](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15).
* OpenCL implementations ship with your vendors GPU drivers.
* OpenGL implementations ship with your vendors GPU drivers.
* SFML pre-built binaries can be obtained from the projects [downloads section](https://www.sfml-dev.org/download.php). Visual C++ 14 (2015) are built using platform toolset v140, which is binary compatible with toolset v141 used by VS 15 (2017) products.
    * After installing, create a (user-level is sufficient) environmental variable with the name SFML_ROOT holding the path to the SFML install root directory. This is so that the FindSFML.cmake scripts that ship with SFML find the installation properly.
* GLEW binaries can be obtained from [here](http://glew.sourceforge.net/).
* GLM can be cloned from the [project repository](https://github.com/g-truc/glm).
    * Building GLM from source is essentially just a copy of the headers with proper CMake configuration and packageConfig.cmake file generation. (Makes it easier to depend on GLM.)
* Qt Open-Source binary installers can be obtained from [here](https://www.qt.io/download-open-source/?hsCtaTracking=f977210e-de67-475f-a32b-65cec207fd03%7Cd62710cd-e1db-46aa-8d4d-2f1c1ffdacea#section-2).
#### Building
* To build from the command-line, open a Visual Studio Command-Prompt (a console with the compiler in the path and all other neccessary environmental variables set up) and type:
```powershell
cmake.exe -G"Visual Studio 15 2017 Win64" \
          -DCMAKE_MODULE_PATH=<path_to_SFML_install_root>/cmake/Modules \
          -DGLEW_INCLUDE_DIR=<path_to_GLEW_install_root>/include \
          -DGLEW_LIBRARY=<path_to_GLEW_install_root>/lib/Release/x64/glew32.lib \
          <path_to_samples_root>
```
* To build using Visual Studio 15, use the _Open Folder_ calapbility. Point Visual Studio to the root directory of the samples. In the CMake context menu, select _Change CMake Settings_ and modify the .json file to look something like this. (Modify in trivial places to match your installation layouts.)
```json
"configurations": [
        {
            "name": "x64-Debug",
            "generator": "Visual Studio 15 2017 Win64",
            "configurationType" : "Debug",
            "buildRoot":  "${env.LOCALAPPDATA}\\CMakeBuild\\${workspaceHash}\\build\\${name}",
            "cmakeCommandArgs": "",
            "variables": [
                {
                    "name": "CMAKE_MODULE_PATH",
                    "value": "C:\\Kellekek\\SFML\\2.4.2\\cmake\\Modules"
                },
                {
                    "name": "GLEW_INCLUDE_DIR",
                    "value": "C:\\Kellekek\\GLEW\\2.0.0\\include"
                },
                {
                    "name": "GLEW_LIBRARY",
                    "value": "C:\\Kellekek\\GLEW\\2.0.0\\lib\\Release\\x64\\glew32.lib"
                }
            ],
            "buildCommandArgs": "-m -v:minimal"
        }
```

### Ubuntu 16.04
#### Installation
Stock Ubuntu packages are sufficient to building the samples
```bash
sudo apt install cmake \
                 ocl-icd-opencl-dev clinfo libclfft-dev libclblas-dev \
                 libsfml-dev libglew-dev libglm-dev \
                 libqt5opengl5-dev
```
#### Building
To build the samples, navigate to the directory where you would like to build the samples and type:
```bash
cmake -DCMAKE_MODULE_PATH=/usr/share/SFML/cmake/Modules/ \
      <path_to_samples_root>
```