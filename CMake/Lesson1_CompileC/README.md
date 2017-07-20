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
- so generators can create nicer IDE project files

## The `add_library` command

