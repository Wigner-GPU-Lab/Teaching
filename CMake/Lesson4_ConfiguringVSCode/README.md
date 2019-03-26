# Lesson 4 - Configuring Visual Studio Code to use CMake


Visual Studio is a cross platform Integrated Development Environment (IDE) suitable for tasks ranging from text editing to compiling and debugging lots of different languages.

## Installing Visual Studio Code

Visual Studio Code is available from [here](https://code.visualstudio.com/Download).

### Extensions for C++ development with CMake

Language support is brought into Visual Studio Code via extensions. For C++ development we recommend the Microsoft C/C++ extension. For CMake support we recommend the CMake extension (by twxs) and the CMake Tools extension (by vector-of-bool).

## C++ Compilers

#### Linux/Unix
Under *nix systems, clang and gcc are the recommended free C++ compilers, you should be able to install them from your package manager (e.g. sudo apt-get install clang) for most of the distributions, or use the prebuilt binaries, or compile from source.


[Clang releases](http://releases.llvm.org/download.html)


[GCC releases](https://gcc.gnu.org/releases.html)

#### Windows
##### The Microsoft Visual C++ compiler
Under Microsoft Windows the native compiler toolchain is provided under the Visual Studio brand (the older, larger brother of Visual Studio Code). You can either install just the so called [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads) (scroll down to the Tools for Visual Studio 2017, and select Build Tools for Visual Studio) containing just the compilers and not the IDE, or if you plan to use other features of the larger package, you can install the free [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/).

When installing either one, first they install the Visual Studio Installer, where you can select from the languages and tools that you need. Make sure you select C/C++ development or Visual C++ build tools and check for CMake related workloads.

##### Clang on Windows
Clang supports Windows, but it does not yet have a standalone C++ Standard Library implementation on Windows, so it uses the Visual C++ compiler's STL implementation, thus it requires the above tools to fully function. Stable releases of clang and LLVM are avilable [here](releases.llvm.org/download.html) (look for Prebuilt binaries Windows 64-bit), and latest snapshot builds are available from [here](llvm.org/builds).

## Configuring Visual Studio Code
If you do not have CMake separately installed and available from the PATH you need to tell Visual Studio Code where to find cmake.exe.

Visual Studio or its Build Tools bring cmake.exe but hide it in their install directory. Search for cmake.exe in your `C:\Program Files` or `C:\Program Files (x86)` directories. For example for the 2017 version of build tools the location of the CMake executable by default is: `C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMAKE\CMAKE\bin\cmake.exe`. Note this path.

After you have installed the above extensions, you need to update preferences and set the path to CMake.

Open Visual Studio Code, and go to: File->Preferences->Settings. In the search bar type cmake. Under the User Settings tab and under the Extensions dropdown list select CMake configuration. Modify the two textboxes (CMake executable and the CMake generator executable) to contain the above path to cmake.exe. Changes are saved automatically and you can close the tab.

You might need to reload the application by closing and reopening. If you have installed the compilers after you have already configured the CMake extensions, you may want to rescan for C++ compilers. Open the command palette (Ctrl+Shift+P), type `cmake` and scroll to the command `scan for kits`. After the scan is finished, results are written to the output window. If everything is fine, it should find the Visual Studio Community kit or the Visual Studio Build Tools and/or the Clang kit. On the right pop-ups may ask for permission to use CMake and/or configure the current folder/project for CMake if you have opened a C++ folder already, you can approve/say yes to them.

## Troubleshooting
### CMake configuration fails immediately
If the above update of the path to the CMake executable is not enough, you might need to manually edit the configuration json file. Go to: File->Preferences->Settings and in the top-right corner click the {} icon. A new text file titled `settings.json` opens. You might try adding the path to the CMake executable and the Ninja configurator:

```json
{
  "cmake.cmakePath": "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe",
  "cmake.configureArgs": [
    "-DCMAKE_MAKE_PROGRAM=C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\Ninja\\ninja.exe"
  ]
}
```

### Scan for kit does not find a compiler or build tool
If the scan fails to find any or some of the compilers, you might try manually adding the necessary kits. Press Ctrl+Shift+P to open the command palette and search for `CMake: Edit user-local CMake kits`. A new text file titled `cmake-tools-kits.json` opens with the compilers found by the scan on your system. After installing Visual Studio or the Build Tools it should find multiple configurations, like amd64, amd64_arm, amd64_x86, x86, x86_amd64, x86_arm You can keep these (although you most likely dont need anything except the amd64 kit, the others are cross compilers for arm and older 32 bit architectures), but if you would like to add clang, you need to add a new kit, copy the amd64 configuration, and change the name.

Some examples of successful configurations:

```json
{
  "name": "Clang 8.0.0 for MSVC with Visual Studio Community 2017 (amd64)",
  "visualStudio": "VisualStudio.15.0",
  "visualStudioArchitecture": "amd64",
  "compilers": {
    "C": "C:\\Program Files\\LLVM\\bin\\clang-cl.exe",
    "CXX": "C:\\Program Files\\LLVM\\bin\\clang-cl.exe"
  }
}
```

```json
{
  "name": "Visual Studio Community 2017 - amd64",
  "visualStudio": "VisualStudio.15.0",
  "visualStudioArchitecture": "amd64",
  "preferredGenerator": {
    "name": "Visual Studio 15 2017",
    "platform": "x64"
  }
}
```


```json
{
  "name": "GCC for x86_64-w64-mingw32 7.2.0",
  "compilers": {
    "C": "C:\\Program Files\\Haskell Platform\\8.6.3\\mingw\\bin\\x86_64-w64-mingw32-gcc.exe",
    "CXX": "C:\\Program Files\\Haskell Platform\\8.6.3\\mingw\\bin\\x86_64-w64-mingw32-g++.exe"
  }
}
```
Be careful that each kit is in {} brackets and successive kit's brackets are separated by commas. After modifications, save the file and reload Visual Studio Code and rescan for kits.

## Starting a new C++ project
To start a new C++ project from Visual Studio select File->Open folder. Navigate to the location you would like to use, create a folder by right clicking, and finally select the desired folder.

To configure, open the command palette: Ctrl+Shift+P, type CMAKE and select Quick Start. The kit selection opens, choose the desired configuration, like Visual Studio ... amd64 or Clang. CMake may asks to collect usage data, you may say No. Enter the name for the new project and in the next question, select executable. After these steps a CMakeLists.txt and a main.cpp file is created in the folder.

<br><br>

----------------------------------------------------------------------------------------------------
⏪ [Prior Lesson](../Lesson3_Dependencies/)
         ⏫ [Back to Top](../)
