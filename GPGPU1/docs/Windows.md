Building on Windows
----------------

We __highly recommend both installing and building 64-bit binaries__ on systems that support it. 32-bit builds on 64-bit enabled operating systems is considered cross-compilation and should be avoided if possible. Search for _x86_64_, _x64_ or _amd64_ markers when installing dependencies, all referring to the same architecture, most often referred to as _64-bit_. This is especially true when building SYCL applications, because the ComputeCpp SDK never had 32-bit version to begin with.

_NOTE: In the coming sections, whenever inside a console invocation something `<is_placed_in_angle_brackets>`, that denotes customization points which you must specify depending on where you installed your dependencies._

## Installing dependencies

Installing dependencies can be done in a few ways (unfortunately). The old-school "go to the website of every project and fetch source/installers" and in some new, semi-automated ways.

### Old-school

* **CMake:** Grab the latest "next-next-finish" installer from the Kitware [downloads section](https://cmake.org/download/).
    * For conveniant usage, either let the installer add cmake.exe to your PATH, or add itt manually.
    * If you plan on building inside Visual Studio, you can skip this step, because a CMake fork comes bundled with Visual Studio if you install CMake support (which you should).
* **C++11/14 compiler:** We highly recommend building native executables using either the Microsoft Visual C++ toolchain (aka. MSVC) or the LLVM/Clang toolchain.
    * **Visual C++:** There are two recommended ways to install it:
        * When used with the Visual Studio IDE, install Visual Studio 2017 Community Edition (for OSS and small team development) from [here](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=Community&rel=15). Because of the still evolving tooling around CMake Server Mode, it is recommended to always install and update to the latest Visual Studio version available.
            * Bleeding edge, preview versions can be obtained from [here](https://www.visualstudio.com/vs/preview/). Allows for side-by-side installation with production versions.
            * Visual Studio ships with a fork of CMake that drives the IDE experience, though having a more recent seperate install can come in handy.
        * When used from the command-line or through another IDE, one can obtain the `Build Tools for Visual Studio 2017` from [here](https://www.visualstudio.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15).
    * **LLVM Clang:** Grab the latest "next-next-finish" installer from the LLVM project [downloads section](releases.llvm.org/download.html). _(Dubbed "Clang for Windows (64-bit)")_
        * If you have Visual Studio already installed, it will also install IDE integration as an applicable toolchain for building C/C++ programs.
        * When prompted to add LLVM tools to your PATH, if you do, CMake will have an easier way finding it. Otherwise it must be specified on the command-line or setup your IDE to provide CMake with the LLVM install location.
* **OpenCL:** implementations ship with your vendors GPU drivers. Usually nothing has to be done, Windows Update installs this for you if you do not do so manually.
    * **OpenCL SDK:** can be obtained in multiple ways:
        * The minimal set of development files can be obtained from the [GPUOpen project](https://gpuopen.com/) via their Github [release section](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)
        * More comprehensive installs with samples and utilities are found on vendor websites:
            * [AMD APP SDK](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)
            * [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
            * [Intel OpenCL SDK](https://software.intel.com/en-us/intel-opencl)
* **clFFT:** can be installed in one of two ways
    * Fetch a pre-built binary from the project [releases section](https://github.com/clMathLibraries/clFFT/releases).
    * Build it from source via CMake
* **clBAS:** can be installed in one of two ways
    * Fetch a pre-built binary from the project [releases section](https://github.com/clMathLibraries/clBLAS/releases).
    * Build it from source via CMake
        * Navigate to the directory where you wish to build
        * `cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<where_you_want_to_install> -DBUILD_SHARED_LIBS=OFF -DBUILD_CLIENT=OFF -DBUILD_SAMPLE=OFF -DBUILD_TEST=OFF -DBUILD_KTEST=OFF <path_to_source_root>/src`
* **SYCL:** There are currently two notable SYCL implementations, the conforming GPU-accelerated ComputeCpp SDK done by Codeplay Ltd. and the non-conforming OpenMP accelerated open-source triSYCL backed by Xilinx.
    * **ComputeCpp:** may be installed from Codeplays website [here](https://www.codeplay.com/products/computesuite/computecpp). _(Community Edition is free, requires registration.)_
    * **triSYCL:** needs no installation, other than cloning the [repo](https://github.com/triSYCL/triSYCL) of the project.
* **OpenGL:** implementations ship with your vendors GPU drivers. Usually nothing has to be done, Windows Update installs this for you if you do not do so manually.
    * **OpenGL SDK:** depends on the windowing library used to build the samples.
        * **SFML:** pre-built binaries can be obtained from the projects [downloads section](https://www.sfml-dev.org/download.php). Visual C++ 14 (VS 2015) are built using platform toolset v140, which is binary compatible with toolset v141 used by VS 15 (VS 2017) products.
            * After installing, create an environmental variable (user-level is sufficient) with the name SFML_ROOT holding the path to the SFML install root directory. This is so that the FindSFML.cmake scripts that ship with SFML find the installation properly.
            * **GLEW:** binaries can be obtained from [here](http://glew.sourceforge.net/).
            * **GLM:** can be cloned from the [project repository](https://github.com/g-truc/glm).
                * Building GLM from source is essentially just a copy of the headers with proper CMake configuration and packageConfig.cmake file generation. (Makes it easier to depend on GLM.)
    * **Qt5:** [Online](https://www.qt.io/download-open-source/#section-2) and [Offline](https://www.qt.io/download-open-source/#section-3) Open-Source binary installers can be obtained from prior links.
        * When selecting install variants, use Desktop OpenGL component, no ANGLE and no OpenGL ES.

### Vcpkg

If you plan on building using MSVC, the Vcpkg project helps in getting many open-source projects to build using Visual C++. It is essentially a peer-reviewed set of CMake scripts and patches to build libraries that originally do not support CMake and/or MSVC.

The list of packages available to be built using Vcpkg are (at the time of writing):

* GLM
* GLEW
* SFML

#### Installing Vcpkg

* Clone its [repository](https://github.com/Microsoft/vcpkg) and bootstrap for future use.
* Set environmental variables for convenient use:
    * `VCPKG_DEFAULT_TRIPLET` to the default build style (recommended value: `x64-windows-static`)
    * `VCPKG_ROOT` to the root of Git repo clone
* From a VS developer command prompt, launch `bootstrap-vcpkg.bat`

#### Building dependencies using Vcpkg

To build and install packages, issue from the repo clone root:

```Powershell
.\vcpkg.exe install sfml glew glm
```

## Registering packages

The list of _good guys_ consist of dependencies that support CMakes [package configuration](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html) files, which are arguably the most portable and convenient way to delpoy libraries to be consumed by downstream CMake projects.

* clFFT
* clBLAS
* GLM
* Qt5

In order to register these packages into the users CMake package registry, open a PowerShell console (pre-installed on all Windows machines) and issue the following commands:

```Powershell
New-Item -Force HKCU:\Software\Kitware\CMake\Packages # If it's the first time you install a package ever. Force creates all missing items recursively
New-Item HKCU:\Software\Kitware\CMake\Packages\clFFT | New-ItemProperty -Name '<something_unique_perhaps_version>' -Value '<path_to_clFFT_install_root>\CMake'
New-Item HKCU:\Software\Kitware\CMake\Packages\clBLAS | New-ItemProperty -Name '<something_unique_perhaps_version>' -Value '<path_to_clBLAS_install_root>\CMake'
New-Item HKCU:\Software\Kitware\CMake\Packages\GLM | New-ItemProperty -Name '<something_unique_perhaps_version>' -Value '<path_to_GLM_install_root>\cmake'
New-Item HKCU:\Software\Kitware\CMake\Packages\Qt5 | New-ItemProperty -Name '<something_unique_perhaps_version>' -Value '<path_to_Qt5_install_root>\<qt_ver_eg_5.9.2>\msvc2017_64\lib\cmake\Qt5'
```

### Vcpkg toolchain

Any of the packages built via Vcpkg need not be registered, but instead can be loaded via a [toolchain file](https://cmake.org/cmake/help/latest/variable/CMAKE_TOOLCHAIN_FILE.html?highlight=cmake_toolchain_file). This is a file loaded very early on in the configuration phase of running CMake, thus circumventing many of the mechanisms that detect default values of variables such as compilers and linkers to use, but can also be used to preset package/library locations.

Adding `-DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg_repo>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static` to the command-line invocation of CMake will add all the neccessary variables for scripts to detect dependencies.

## Building

The means of building depends on what you installed: which compiler and what IDE (if any). The installation methods described aimed to minimalize the size of cmake invocations, but some dependencies require special handling.

### Command-line

```powershell
cmake.exe -G"Visual Studio 15 2017 Win64" \
          -DCMAKE_MODULE_PATH=<path_to_SFML_install_root>/cmake/Modules \
          -DGLEW_INCLUDE_DIR=<path_to_GLEW_install_root>/include \
          -DGLEW_LIBRARY=<path_to_GLEW_install_root>/lib/Release/x64/glew32.lib \
          -DCOMPUTECPP_PACKAGE_ROOT_DIR=<path_to_ComputeCpp_install_root> \
          <path_to_samples_root>
```

#### MSVC

Placing the MSVC compiler onto the PATH is not enough to make the compiler work. It also requires several environmental variables. To make it easier to invoke the compiler, from the command-line, its installer creates a Start Menu entry labeled `x64 Native Tools Command Prompt for VS 2017`. Essentially, this opens a `cmd.exe` console invoking `vcvars64.bat`, which actually sets up the environment for the compiler.

#### LLVM

If you let the LLVM installer add it to the PATH, CMake will always find it automatically. Otherwise, it must be instructed via the `-DCMAKE_C_COMPILER=<path_to_clang.exe>` and `-DCMAKE_CXX_COMPILER=<path_to_clang++.exe>` respectively.

### Visual Studio 2017

To build using Visual Studio 15, use the _Open Folder_ capability. Point Visual Studio to the root directory of the samples. In the CMake context menu, select _Change CMake Settings_ and modify the .json file to look something like this. (Modify in trivial places to match your installation layouts.)

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
                    "value": "<path_to_GLEW_install_root>\\cmake\\Modules"
                },
                {
                    "name": "GLEW_INCLUDE_DIR",
                    "value": "<path_to_GLEW_install_root>\\2.0.0\\include"
                },
                {
                    "name": "GLEW_LIBRARY",
                    "value": "<path_to_GLEW_install_root>\\lib\\Release\\x64\\glew32.lib"
                },
                {
                    "name": "COMPUTECPP_PACKAGE_ROOT_DIR",
                    "value": "<path_to_ComputeCpp_install_root>"
                }
            ],
            "buildCommandArgs": "-m -v:minimal"
        }
```