# Lesson 1 - Compile C


Perhaps the sipmlest compilation process is to compile a single C file (though C++ isn't any more complicated). How is this done?

```CMake
add_executable(Lesson1 Main.c)
```

Simple, right? In fact, this single line of code without anything else will get the job done. What does this command do?

## The `add_executable` command

The `add_executable` command instructs CMake to create an executable program from one of the languages that are built into CMake. These languages in the best and latest (at the time of writing) CMake 3.9 are:

- Fortran
- C
- C++
- C#
- CUDA

Other languages may also be compiled fairly easily (with external "library" support, more on this later), but the built-in `add_executable` command understands the compilation model of these languages. The command selects the language used based on file extensions.

If a program consists of multiple source files, no worries, just list them all one after the other.

```CMake
add_executable(MyExec src/this.c src/and.c src/that.c)
```

If your sources get too many in number, you can place an arbitrary number of spaces, tabs, line breaks in between function arguments to tidy up your scripts.

```CMake
add_executable(MyExec src/one.c
                      src/too.c
                      src/many.c
                      src/sources.c
                      src/for.c
                      src/a.c
                      src/single.c
                      src/line.c)
```

If you want to group your script by declaring your sources in advance, just stick them into a variable

```CMake
set(SOURCE_FILES src/one.c
                 src/two.c)

add_executable(MyExec ${SOURCE_FILES})
```

### Include files and directories

In C/C++, the canonical spearation of function declarations and definitions is done via header and source files. One translation unit is the code span that one invocation of the compiler sees. In between translation units, the compiler "forgets" everything.

Let's say we have a one source and one header file in our program layed out the following way:

```
my_project -+
            |
            Header.h
            |
            Source.c
            |
            CMakeLists.txt
```

With contents

#### Header.h

```C
#include <stdio.h>

void greet() { printf("Hello\n"); }
```

#### Source.c

```C
#include "Header.h"

int main()
{
        greet();

        return 0;
}
```

#### CMakeLists.txt

```CMake
project(Lesson1 LANGUAGES C)

add_executable(Lesson1 Source.c)
```

_NOTE: the vigilant reader might have noticed, that we defined a function in a header file just a few lines after explaining how it violates the header/source distinction. This server the sole purpose of proving a point in just a moment._

Now, if we build this project, this is what we'll see:

```
$ cmake -G "NMake Makefiles" ..\
-- The C compiler identification is MSVC 19.11.25505.0
-- Check for working C compiler: .../cl.exe
-- Check for working C compiler: .../cl.exe -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Configuring done
-- Generating done
-- Build files have been written to: ...
$ cmake --build .
Scanning dependencies of target Lesson1
[ 50%] Building C object CMakeFiles/Lesson1.dir/Source.c.obj
Source.c
[100%] Linking C executable Lesson1.exe
[100%] Built target Lesson1
$ .\Lesson1.exe
Hello
```

A few things to note:

- We included the C-runtime `stdio.h` using `#include <stdio.h>` with angle brackets, whereas our custom `Header.h` was included using `#include "Header.h"` with quotation marks.
  - The angle bracket notation searches for the file in an implementation-defined set of folders, plus user-defined paths set on the command-line. C-runtime and most system headers need not be specified manually.
  - The quote notation first searches in the same directory as the source file being compiled before falling back to the angle bracket notation.
- Even though we specified `Source.c` as the only input to our executable, it still compiled. Including `Header.h` was done by the compiler, not CMake.

Now, having built our application, if we were to modify `Header.h` and change the message per se to read "Helloooo", we might expect our build system to do nothing, because we have not touched `Source.c`, the only source file presented to CMake. On the contrary, our application is rebuilt. Why?

CMake has an understanding of the C/C++ compilation model and generates makefiles that not only operate on the timestamps of the input files, but also carry out automatic dependency tracking assisted by the compilers. You might have noticed the

>Scanning dependencies of target Lesson1

Part in the console output when we built our project. This is the step where the dependency detection part of the generated makefiles are processed. The compiler is invoked with flags that trigger the generation of a database file holding all `#include` files. In turn, timestamps of these files are also checked to decide, whether our `Source.c` needs recompilation.

### Explicit intra-target dependencies

While everything above works, it is good practice to not rely on this feature inside a target. Header files belonging to a target should be explicitly mentioned:

- to facilitate understanding the dependencies within your project,
- so generators can create nicer IDE project files.

Automatic dependency tracking is not an excuse to omit header file definitions inside the build scripts. Its purpose becomes apparent when one starts using libraries. Declaring your headers is as simple as declaring the sources were. CMake will know that files with header extensions are not to be compiled separately.

```CMake
add_executable(Lesson1 Header.h Source.c)
```

## The `target_include_directories` command

Often one will see that even medium sized projects are organized in a similar manner:

```
my_project -+
            |
            inc -+
            |    |
            |    Header1.h
            |    |
            |    Header2.h
            |    |
            |    ...
            |
            src -+
            |    |
            |    Source1.c
            |    |
            |    Source2.c
            |    |
            |    ...
            |
            CMakeLists.txt
            |
            README.md
```

This might seem silly at first, but very soon one will start to have many types of files: headers, sources, documentation fragments, GPU shaders, GPGPU kernels, makefiles, etc. In such cases, including the headers from the sources might look like this:

```C
#include "../inc/Header1.h"
```

Not only is this ugly, but very error-prone to moving files around to have a meaningful layout on disk. One not only has to update the build scripts when files are moved, but also the source files.

To overcome this, one may instruct CMake (and thus the compiler) to append the `inc` directory to the angle bracket include search paths. To do this, one might say:

```CMake
add_executable(example inc/Header1.h
                       inc/Header2.h
                       src/Source1.c
                       src/Source2.c)

target_include_directories(example PRIVATE inc)
```

Doing so, one is allowed to include them in code like this:

```C
#include <Header1.h>
```

>The `target_include_directories` command essentially controls the `-I` or `/I` switch of the compiler invocation.

This (at least in my experience) is good practice and pays off to get used to. The `PRIVATE` argument is one of three possible choices as explained in the [commands docs](https://cmake.org/cmake/help/latest/command/target_include_directories.html?highlight=target_include_directories), which control the propagation of this property to downstreams, consumers of the target.

- `PRIVATE` is used when consuming targets need not inherit the given property. In the case of an executable, this is safe to use, as nobody can "link" to an executable. In the case of libraries, this might be the case when an include need only be inside source files, but not headers. This is the case when a library depends on another but only, as an implementation detail. The dependants types do not show up in the interface of the library being compiled. Thus downstreams need not append the include directory to their own.
- `PUBLILC` is used when consuming targets need to inherit the given property. If types manifest in the interface of the library, or definitions need to surface, downstreams must be able to locate the header files used, but not owned by the library.
- `INTERFACE` is used on "meta-targets", libraries that do not produce actual code, but instead are just a collection of properties that consumers must know about. One prime example is OpenMP, which may very well be implemented inside the C-runtime library, so no extra linkage may be required, but compile flags are required to turn on OpenMP support. _(The `INTERFACE` argument is not exclusive to target_include_directories, more on this later.)_ In this case "linking" against an interface library populates corresponding properties of the target recorded on the interface llibrary recorded via `INTERFACE`. _NOTE: OpenMP detection ships with CMake and isn't defined by the user, thus not only is it an interface library, but also an `IMPORTED` library. More on imported targets later._

## The `add_library` command

Let's say that we either want to build a library, either as an end product, or simply to factor out some of the code that will be reused by multiple executables.

```CMake
add_library(util Functions.h Functions.c)

add_executable(use1 Use1.c)
add_executable(use2 Use2.c)

target_link_libraries(use1 PUBLIC util)
```