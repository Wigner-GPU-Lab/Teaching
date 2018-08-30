# Lesson 3 - Dependencies


One of the strengths of CMake is its ability to find dependencies with minimal user effort (based on
upstream support). There are two categories of dependencies, both with their preferred method of
detection:

- Upstream is a binary or is built using another technology
- Upstream was built using CMake

There is also a hidden third option:

- Upstream is built using another technology, but ships with CMake support

## Find Module dependency detection

Find module scripts target upstreams that are not CMake aware, meaning that they either ship as a
binary or are built using other build systems. The layout of such libraries/tools are too numerous
to count, not to mention their install locations may be unique, or worse: user defined. These
scripts try looking for libraries in typical install locations on the OS at hand. Some libraries may
create environmental variables when installed properly, which may guide the find module scripts.
Guessing user defined install locations in advance requires otherwordly support. Depending on such
libraries will need minimal user interaction.

The scripts that detect such dependencies are called _Find Module scripts_, and are named by convention
as `FindName.cmake`, where "Name" is the name of the dependency, for example. `FindMPI.cmake`.
Authoring such scripts is outside the scope of this lesson, but will be explained in a later lesson.
For the time being, we'll restrict ourselves to browsing the comments section of these scripts.

CMake (3.9) comes with 148 pre-installed find module scripts. _(On Windows they are located under
`<install root>\share\cmake-3.9\Modules` and on Ubuntu 16.04 under
`/etc/share/usr/share/cmake-3.5/Modules/`)_ You might want to scan through the list to get a feeling
of what comes bundled with CMake. Let's take a look at FindMPI.cmake, shall we?

```rst
# FindMPI
# -------
#
# Find a Message Passing Interface (MPI) implementation
#
# The Message Passing Interface (MPI) is a library used to write
# high-performance distributed-memory parallel applications, and is
# typically deployed on a cluster.  MPI is a standard interface (defined
# by the MPI forum) for which many implementations are available.
#
# Variables
# ^^^^^^^^^
#
# This module will set the following variables per language in your
# project, where ``<lang>`` is one of C, CXX, or Fortran:
#
# ``MPI_<lang>_FOUND``
#   Variable indicating the MPI settings for ``<lang>`` were found.
# ``MPI_<lang>_COMPILER``
#   MPI Compiler wrapper for ``<lang>``.
# ``MPI_<lang>_COMPILE_FLAGS``
#   Compilation flags for MPI programs, separated by spaces.
#   This is *not* a :ref:`;-list <CMake Language Lists>`.
# ``MPI_<lang>_INCLUDE_PATH``
#   Include path(s) for MPI header.
# ``MPI_<lang>_LINK_FLAGS``
#   Linker flags for MPI programs.
# ``MPI_<lang>_LIBRARIES``
#   All libraries to link MPI programs against.
```

Up to this point, the comments section demonstrates how to depend on MPI using the "old school"
method. The comments are fairly straightforward: once the scripts are run, they set certain
variables which we may use in our build scripts.

How do we run these find module scripts? We use the `find_package` command like this:

```CMake
find_package(MPI REQUIRED)

add_executable(app Main.c)

target_include_directories(app PRIVATE ${MPI_C_INCLUDE_PATH})

target_link_libraries(app PRIVATE ${MPI_C_LIBRARIES})

set_target_properties(app PROPERTIES LINK_FLAGS ${MPI_C_LINK_FLAGS}
                                     COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
```

If you think to yourself, this seems like a great deal of work only to link to MPI, you are right.
(Hold on a sec, we'll have you covered.) All this is required because there are quite a few MPI
implementations out in the wild, and `FindMPI.cmake` knows how to use just about all of them. Some
require just include directories set up, some require only linking, others require extra compile
definitions... even you you know for a fact that the MPI flavor you have installed only requires
linkage _per se_ (in which case variables like `${MPI_C_COMPILE_FLAGS}` are empty), others have gone
to great lengths in order to collect all the MPI variations and the means of consuming them.
Sometimes it's nice to just not think about stuff and do as one's told and rest assured that it's
always going to be fine.

### Import target syntax

If you (dear reader) were one of the few who thought, that is a lot of typing, let's read on the
comments section:

```rst
# Additionally, the following :prop_tgt:`IMPORTED` targets are defined:
#
# ``MPI::MPI_<lang>``
#   Target for using MPI from ``<lang>``.
```

An IMPORTED target is a library that is not meant to be built, but only collects certain properties
to be inherited by other targets. This comes in handy right now (and was created for just this
case), because the MPI library is already built, we have no work with it; we would only like to
inherit a set of compiler and linker options.

So how do we use an imported target?

```CMake
find_package(MPI REQUIRED)

add_executable(app Main.c)

target_link_libraries(app PRIVATE MPI::MPI_C)
```

Wow! That is much simpler. All the properties we set manually earlier are now all inherited from the
imported target. The first part of imported target, before `::` is always the name of the module at
hand. The second part are "sub-modules", parts of the module than can stand by themselves. In the
case of MPI, an implementation may omit Fortran bindings for example, but that doesn't mean we
couldn't rely on the C bindings.

Another prime example is Boost, with many subprojects which may be consumed separately using
`Boost::filesystem`. In this case, one need not detect all subprojects (they might even be absent).

```CMake
find_package(Boost REQUIRED VERSION 1.56
                            COMPONENTS filesystem)

add_executable(app Main.cpp)

target_link_libraries(app PRIVATE Boost::filesystem)
```

One can not only request the minimal set of submodules, but only the minimal version accepted.

Unfortunately, not all modules provide the imported target syntax. Should you take interest in
depending on such a module, and you feel like contributing "for the greater good", it is fairly
simple to cook up imported target syntax based on any one of the other modules that support it.
Posting your patches to the cmake-developer mailing list, you could make the lives of others easier
as well.

Following parts of the comments section of `FindMPI.cmake` are useful for building custom build
targets. These will be explained in a coming lesson of this tutorial.

### Shipping Find Module scripts

You may find yourself in a place where you'd want to ship a find module script along with your
program. For example:

- The find module scripts shipping with a given CMake version you wish to target are too old, and
  you want to ship a more up-to-date version.
- The find module scripts do not ship with CMake, but the author of the library provides one (SFML
  for example).
- Neither CMake, neither the library author provides such a script, you just found one online,
  written by a desperate soul.

In such cases, it is customary to ship such scripts in a folder layout like this:

```
my_project -+
            |
            cmake -+
            |      |
            |      Modules -+
            |               |
            |               FindSFML.cmake
            |
            inc -+
            |    |
            |    Header1.hpp
            |    |
            |    ...
            |
            src -+
            |    |
            |    Source1.cpp
            |    |
            |    ...
            |
            CMakeLists.txt
            |
            ...
```

## Package Config dependency detection

When an upstream was built using CMake, it may also create a set of scripts that are tailored to the
given install, no matter how exotic. As a result, finding and depending on such libraries always
succeeds — 100% of the time, with zero user interaction required. These scripts are referred to as
Package Config scripts.

One such library for instance is [clFFT], an OpenCL accelerated FFT library.

When using such libraries, one usually has to build the library from source. Of course, package
config type dependencies are available when the upstream is __built__ with cmake.

Let's say one does the follwing (or equivalent):

```
git clone https://github.com/clMathLibraries/clFFT.git
cd clFFT
git checkout v2.12.2
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/opt/clMath/clFFT/2.12.2 ../src
cmake --build . --target install
```

This will fetch the latest (at the time of writing) tagged version of clFFT, build it and install it
under the user's home directory. _(NOTE: `CMAKE_INSTALL_PREFIX` is the canonical variable that
controlls the target location of the `install` target. More on magic parameters like this later.)_

So where are the magic package config scripts? After taking a look at the install we'll find a
folder:

```
$ ls ~/opt/clMath/clFFT/2.12.2/CMake
clFFTConfig.cmake
clFFTConfigVersion.cmake
clFFTTargets.cmake
clFFTTargets-debug.cmake
```

## Guiding Find Module and Package Config scripts

### `CMAKE_MODULE_PATH`

When shipping Find Module scripts in non-system locations, one may see error messages containing the following when using the following snippet:

**Snippet**
```CMake
find_package(MyFavoriteLib REQUIRED)
```

**Error**
```
CMake Error at CMakeLists.txt:1 (find_package):
  By not providing "FindMyFavoriteLib.cmake" in CMAKE_MODULE_PATH this
  project has asked CMake to find a package configuration file provided by
  "MyFavoriteLib", but CMake did not find one.
```

The error message is pretty self explanatory. CMake by default looks in its own installation path for Find Module scripts. A few ship with CMake itself, such as the earlier mentioned `FindMPI.cmake`. If one has to ship such a script, or is installed in a location that CMake is unaware of, one has to add it to the `CMAKE_MODULE_PATH` variable. If it's located in the canonical location as suggested in the earlier section, one may do the following:

```CMake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
```

_(This snippet assumes the root `CMakeLists.txt` file contains a project statement and the script engine has not encountered another one up until this statement. See later for a set of do's and don'ts in the Modern CMake style guide.)_

### `CMAKE_PREFIX_PATH`

In our previous error message, we have omitted its other half, which reads as:

**Error**
```
  Could not find a package configuration file provided by "MyFavoriteLib"
  with any of the following names:

    MyFavoriteLibConfig.cmake
    myfavoritelib-config.cmake

  Add the installation prefix of "MyFavoriteLib" to CMAKE_PREFIX_PATH or set
  "MyFavoriteLib_DIR" to a directory containing one of the above files.  If
  "MyFavoriteLib" provides a separate development package or SDK, be sure it
  has been installed.
```

Once again, we need not think much on the fix. If we intended on using a Package Config script instead of a Find Module one, CMake is going to look for scripts named in a given convention that differs from the previous convention.

Because Package Config scripts are typically found in the installation tree of the given library, it is most likely unique to every machine. Therefore it is not wise to bake the fix into the build scripts, as they will not be portable. Instead, we provide the addition on the command-line with the `-D` switch, which allows us to define variables when invoking CMake, instead of using `set()` inside the scripts.

```
cmake -D CMAKE_PREFIX_PATH=/comma/semi-colon;/list/of/folders;/which/contain/config/scripts
```

_(One may recall, that in CMake everything is a string. Earlier, we used the `list(APPEND)` function to append to `CMAKE_MODULE_PATH`. Actually, CMake lists as [noted in the docs] are nothing more, than a single, semi-colon delimited string. This manifests in the previous invocation.)_

### Package registry

Even with a moderate number of packages used for a build, the length of the command-line invocation can rapidly grow to unwieldy lengths. In order to keep this problem at bay, one may configure an IDE to always add these paths (more on this later in the IDE lesson), or one may register these folders in CMake's [package registry].

Some well written CMake scripts may even register the installations by themselves, though generally
this task remains an exercise for the user. (More on this later in the package authoring lesson.)

#### Ubuntu

On *nix derivates, CMake will inspect the folder `~/.cmake/packages`. In it, it will look for
folders with the same name as the package, and inside it will look for plain text files of arbitrary
names that contain the path to the package config scripts.

```
mkdir ~/.cmake
mkdir ~/.cmake/packages
mkdir ~/.cmake/packages/clFFT
echo ~/opt/clMath/clFFT/2.12.2/CMake > ~/.cmake/packages/clFFT/2.12.2
```

The name of the file need not be the version of the package, but anything that holds information to
you. To list all the installed packages, just recursively list the packages folder.

```
ls -R ~/.cmake/packages
```

#### Windows

On Windows, CMake will inspect the user's registry database, which is most easily edited using
Powershell. Let's say we installed clFFT under `C:\Program Files\clMath\clFFT\2.12.2`

```Powershell
New-Item HKCU:\SOFTWARE\Kitware\CMake\Packages
New-Item HKCU:\SOFTWARE\Kitware\CMake\Packages\clFFT
New-ItemProperty -Path HKCU:\SOFTWARE\Kitware\CMake\Packages\clFFT -Name 2.12.2 -Value 'C:\Program Files\clMath\clFFT\2.12.2\CMake'
```

For those unfamiliar with Powershell, `HKCU:\` is a PSDrive (Powershell Drive, a traversible virtual
drive for tree-like structures) for the system registry; more precisely it is the user's own
registry. (It's an abbeviation of _Hive Key Current User_.) Again, the name of the property need not be the
same as the version string, just something that holds meaning preferably. To list all the installed
packages, just list the packages registry folder.

```Powershell
Get-ChildItem HKCU:\SOFTWARE\Kitware\CMake\Packages
```

_NOTE: on Windows, packages can be installed system-wide for all users when registered into
`HKLM:\`, the local machine's registry, as opposed to the user's registry. This requires
administrator priviliges._

### Using packages

After doing so, every project that uses clFFT can link to it like so:

```CMake
find_package(clFFT REQUIRED)

add_executable(app Main.cpp)

target_link_libraries(app PRIVATE clFFT)
```

Note that again, we didn't have to specify any include directories, because clFFT advertises this
information to downstreams.

### Non-CMake upstreams with Package Config

The distinction of an upstream providing CMake Package Config files is notable because it requires
code generation tools which assemble the config files, which is a considerable amount of effort.
Perhaps the most notable such framework is the set of Qt5 libraries.

Qt5 when installed not only installs Package Config scripts along with the installation _(although
in 5.9 it still omits registering them in the appropriate package registry)_, but Qt developers have
cooked some tooling invocation natively into CMake.

Qt5 uses a few extensions to the C++ language guided by a set of tools that do additional
"compilation" steps upon building, resulting in extra source files that need compiling and linking
to the target. The most basic such tool takes care of the MOC (Meta Object Creation) step. The
`moc.exe` tool inspects the source files, looks for a magic define, and if it's found, it emits an
extra source file which the C++ compiler also needs to compile and link.

This process is totally automated without just about any user interaction.

```CMake
find_package(Qt5 REQUIRED COMPONENTS Core)

add_executable(App Main.cpp)

target_link_libraries(App PRIVATE Qt5::Core)

set_target_properties(App PROPERTIES AUTOMOC ON)
```

Qt5 contains a few more such extra compilation steps, again guided by native CMake support, but
explaining all of them remain outside the scope of this lesson.

## Toolchains

CMake also has the notion of toolchains, which may be used to control/guide dependency detection. A toolchain in general is a set of executables that are used to transform source code into binaries. _(... roughly speaking.)_ CMake has direct support for controlling on how to pick up such tools from the system via so called [toolchain files].

These toolchain files are loaded fairly early during configuration and although meant for prescribing the use of a given toolset, it may also be provided to provide any set of variables and also to run custom scripts.

_NOTE: [Vcpkg], a cross-platform pacakge manager written partly in CMake (ab)uses this feature to override regular dependency detection and pick up self-hosted builds of 1000+ libraries. This is the single most useful package manager for users of CMake, especially on Windows, which lacks a package manager hosting development packages for myriads of open-source libraries._

To provide a toolchain file, provide the `CMAKE_TOOLCHAIN_FILE` variable when invoking CMake, pointing it to the path to such a file.

[clFFT]: https://github.com/clMathLibraries/clFFT
[noted in the docs]: https://cmake.org/cmake/help/latest/command/list.html#introduction
[package registry]: https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html?highlight=package#package-registry
[toolchain files]: https://cmake.org/cmake/help/latest/variable/CMAKE_TOOLCHAIN_FILE.html?highlight=cmake_toolchain_file
[Vcpkg]: https://github.com/Microsoft/vcpkg

<br><br>

----------------------------------------------------------------------------------------------------
⏪ [Prior Lesson](../Lesson2_LinkC_CPP/)
         ⏫ [Back to Top](../)
