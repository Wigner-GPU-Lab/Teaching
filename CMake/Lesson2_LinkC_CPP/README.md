# Lesson 2 - Link C/C++


Compiling a standalone executable is nice, though more often than not is not enough. One might wish to rely on external code compiled into a library and link to it. Buiding a library and linking against it comes by invoking two simple commands: `add_library` and `target_link_libraries`.

```CMake
add_library(util Util.h Util.c)

add_executable(app Main.c)

target_link_libraries(app PRIVATE util)
```

Should one be confused what's going on here, all will be made clear momentarily.

## The `add_library` command

Let's say that we either want to build a library, either as an end product, or simply to factor out some of the code that will be reused by multiple executables.

```CMake
add_library(util inc/Functions.h
                 src/Functions.c)

target_include_directories(util PUBLIC inc)
```

The `add_library` command instructs CMake to emit a library of the defult type. Taking a look at [the docs](https://cmake.org/cmake/help/latest/command/add_library.html?highlight=add_library), we can see that there is an optional param which may override the default library type controlled via `BUILD_SHARED_LIBS` boolean variable.

_NOTE: The actual name of the library on disk is OS dependant. A shared library on *nix-like OSes will be `libname.so` and `name.dll` on Windows, whereas static libraries are named `libname.a` and `name.lib` respectively._

The available library types are:

- `STATIC` will force the creation of a static library.
- `SHARED` will force the creation of a dynamic library.
- `MODULE` will create a dynamic library that no target may link to at compile-time. These libraries are meant to be loaded at run-time.
- `OBJECT` will create an object library. An object library is a group of sources compiled to a single object file (but not a library). You can think of it as a logical grouping of source files. Object libraries like MODULE directories are not linked to, but instead may appear among the source file definitions of another library or executable. For eg. as of CMake 3.9, CUDA is a built-in language and CUDA `.cu` sources may be compiled to an object library (optionally with a special `.ptx` extension as opposed to the default `.obj`. See [CUDA_PTX_COMPILATION](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_PTX_COMPILATION.html?highlight=add_library)). Doing so, one can even use separate compilers for select source files of the same target.

## The `target_link_libraries` command

When a library target has been declared, one can link to it using the following set of commands:

```CMake
add_library(util util/inc/Functions.h
                 util/src/Functions.c)

target_include_directories(util PUBLIC inc)

add_executable(use use/src/Use.c)

target_link_libraries(use PRIVATE util)
```

There are multiple things to note here:

- Nowhere did we specify whether the library that we link to is a static or dynamic. CMake will take care of linkage in both cases.
  - If the library is compiled to be dynamic, by default it will be placed in the same directory where executables are placed, so they can find them conveniently by launching them from `./`
- We did not have to redeclare the include directory of `util` on the target `use`. That is because we declared the `inc` directory to be `PUBLIC`, thus it is propagated to all consumers of `util`.
  - However, the use of `util` itself is totally a private matter of the executable, no further propagation of this linkage information is required. _NOTE: placing `PUBLIC` instead of `PRIVATE` wouldn't hurt either._