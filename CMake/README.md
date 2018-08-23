# CMake tutorial

![alt text](https://www.kitware.com/main/wp-content/uploads/2016/04/CMake_logo.svg)

This repository contains a step-by-step guide that wishes to guide you through understanding CMake,
an increasingly ubiquitous cross-platform build system generator.

----------

## What? Why?

### What is a build system?

Even if the notion of a build system is not clear, you have most likely encountered one or another.
A build system in essence is a tool that takes care of building a compiled language from source code
to a binary with minimal effort from both the user (developer) and the machine executing the build
process. Therefore one does need not care about the actual series of commands to issue to get the
job done and what shortcuts can be taken.

Minimal user effort manifests in authoring a 'makefile', some definition of all the inputs and
outputs of the build process. Whether this is a text file written by the user or is the manifest of
a 'click-click' GUI process depends on the build system at hand.

Traditionally build systems have centered around ahead-of-time compiled languages, such as Fortran,
C/C++ and have relied on file time stamps to detect the minimal set of tasks to execute during a
build (and most still do). A common pattern in the compilation process of all ahead-of-time compiled
languages is that the input to the process is a set of source files on disk and the result is a set
of binary files with a set of intermediate set of binary files, side effects of the build process.
Aside from some exceptionally modern build systems, just about all of them take this process as the
cornerstone of their foundation.

Since their earliest appearence, build systems have learned to become general purpose task execution
engines, which we will soon learn how to put to our advantage.

### What is a build system generator?

A build system generator is a tool that generates makefiles from yet another makefile. Sounds silly,
but doing so one can gain portability, human readability or simply flexibility in doing more complex
tasks.

Modern software compilation and distribution may become arbitrarily complex, especially when one
wishes to take into account the the varying nature of some of the widely available platforms one may
wish to target. Because build systems are often native to a platform or are part of a development
environment that again may be native to a platform, it is useful to create a 'meta-tool' that is
independent of all execution environments and solely takes on the responsibility of creating
makefiles of the desired flavor.

### CMake

CMake is a set of tools that allows to define build processes in a human-readable script as well as
enhancing the process with tasks commonly associated with programming, such as unit testing and
packaging. The CMake scripts are then turned into makefiles of the users choice which act as
execution engines of the CMake scripts.

## How?

### Installation

Installing CMake is as easy as it gets.

#### Linux

Using the package manager of your distribution, fetch pre-built binaries.

Ubuntu:
```
sudo apt install cmake
```
OpenSUSE:
```
yum install cmake
```

#### Windows

Fetch the latest binary from [Kitwares site](https://cmake.org/download/). The MSI installer will
guide you through the 'Next-Next-Finish' process of installing. When prompted for adding CMake to
the system or user PATH, you may want to approve for later convenience. Alternatively, you may
extract the ZIP file manually.

Optionally, you can use [Chocolatey](https://chocolatey.org/):

```
choco install cmake
```

### Verify

To check your installation, you can invoke cmake and query its version

```
user@host:~$ cmake --version
cmake version 3.5.1

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```