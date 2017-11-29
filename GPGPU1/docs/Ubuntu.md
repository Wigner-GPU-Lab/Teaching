Building on Ubuntu 16.04
---------------

Should you use any other distro than Ubuntu, please adapt the following guide to your distribution of choice.

## Installing dependencies

Stock Ubuntu packages are sufficient to build most of the samples

```bash
sudo apt install cmake \
                 ocl-icd-opencl-dev clinfo libclfft-dev libclblas-dev \
                 libsfml-dev libglew-dev libglm-dev \
                 libqt5opengl5-dev
```

The only dependencies missing from the official Ubuntu repository are:

* **C++11/14 compiler:** we highly recommend building using the latest compilers available to your platform, which at the time of writing may be GCC 7 or LLVM/Clang 5.
    * **GCC:** latest GCC compilers are easiest to install via the official toolchain testing private package archive of Canonical.
        ```bash
        sudo add-apt-repository ppa:ubuntu-toolchain-r/test
        sudo apt update
        sudo apt install gcc-7 g++-7
        ```
    * **LLVM:** latest LLVM compiler toolchains are easiest to install via the official apt respositories of the LLVM project. Add the following to a file under `/etc/apt/sources.list.d/` for eg. called `llvm.list`:
        ```
        deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main
        deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial main
        # 4.0
        deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main
        deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-4.0 main
        # 5.0
        deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main
        deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-5.0 main
        ```
        After this, issue:
        ```bash
        wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
        sudo apt update
        sudo apt install clang-5.0 lldb-5.0 lld-5.0
        ```
* **ComputeCpp:** which may be installed from Codeplays website [here](https://www.codeplay.com/products/computesuite/computecpp). _(Community Edition is free, requires registration.)_

## Building

Two typical build scenarios will be demonstrated: building on the command-line and using an IDE, such as [Visual Studio Code](https://code.visualstudio.com/).

As noted [inside the docs](https://code.visualstudio.com/docs/setup/linux), it is easiest to install on Ubuntu via the projects apt repo:

```bash
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo mv microsoft.gpg /etc/apt/trusted.gpg.d/microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code
```

### Command-line

To build the samples, navigate to the directory where you would like to build the samples and type:

```bash
cmake -DCMAKE_MODULE_PATH=/usr/share/SFML/cmake/Modules/ \
      -DCOMPUTECPP_PACKAGE_ROOT_DIR=<path_to_ComputeCpp_install_root> \
      <path_to_samples_root>
```

#### Non-system compilers

When using any compiler other than the default of the OS, CMake must be instructed to use it via specifying two variables: `-DCMAKE_C_COMPILER=<path_to_gcc-7_or_clang-5.0>` and `-DCMAKE_CXX_COMPILER=<path_to_g++-7_or_clang++-5.0>` respectively. When installed via the aforementioned repos, they are already in the PATH.

### Visual Studio Code

To build using Visual Studio Code, we recommend using [Microsofts C/C++ extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) in tandem with [vector-of-bools CMake Tools](https://marketplace.visualstudio.com/items?itemName=vector-of-bool.cmake-tools) and [twxs' CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake) extension. In the File context menu, select _Preferences_ and _Settings_. Add the following entries to

* the user specific json, in case you don't want to repeat this setup for all projects in the future (Vcpkg related entries are prime candidates for user settings)
* the workspace specific json, if you want these variables to persist only for the given project folder (OpenCL and ComputeCpp)

```json
{
    "cmake.configureArgs": [
        // Use latest GCC
        "-DCMAKE_C_COMPILER=<path_to_gcc-7_or_clang-5.0>",
        "-DCMAKE_CXX_COMPILER=<path_to_g++-7_or_clang++-5.0>",
        // SFML entries for FindSFML.cmake
        "-DCMAKE_MODULE_PATH=/usr/share/SFML/cmake/Modules/",
        // ComputeCpp entries for FindComputeCpp.cmake
        "-DCOMPUTECPP_PACKAGE_ROOT_DIR=C:/Kellekek/Codeplay/ComputeCpp/0.3.3"
    ]
}
```