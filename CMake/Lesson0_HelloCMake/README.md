# Lesson 0 - Hello CMake


As in all scripting and programming tutorials, we will begin with a simple, "Hello CMake" script.
While the market value of printing "Hello, World!" is fairly low, it provides an excuse to introduce
many CMake concepts. The script looks like the following:

```CMake
message("Hello, CMake!")
```

Before we could go into detail what's going on in this single line of code, we must first talk a
little about how CMake works. _Much like one would have to give a jump-start to the C/C++
compilation model when confronted with the first "Hello World" in either of the two._

## CMake internals

CMake essentially gets its job done, creating makefiles in the flavor of choice by going through two
steps in order: __configuration__ and __generation__.

### Configure

The configure step is first. This is when CMake parses and processes the CMake scripts the user
provides. Errors in the scripts, whether syntactic or logical, will produce configuration-time
errors which more often than not will halt the processing of the CMake scripts, aborting the entire
process. Some logical errors that CMake cannot forsee (much like run-time errors in all programming
languages) will create errors during the build process.

### Generation

Once the scripts have been processed and no errors have been encountered, CMake enters the makefile
generation phase. This is the time when the set of processed CMake commands are turned into
makefiles, which are written to disk.

Do not be alarmed by the amount of makefiles generated. They are logically equivalent to whatever
was in the CMake scripts. This complexity allows them to execute quickly and produce nice,
human-readable output. They are __generated files__ and are __not meant for editing__ (or even
understanding).

## CMake scripting language

Perhaps the most daunting part of CMake is its scripting language. Compared to modern script
languages, CMake has a fairly simple, unexpressive scripting language. The reason for its
simplicity is that it is the common denominator of all build systems. It does not wish to surface
features which are complicated (or impossible) to implement in any back-end.

Plus, a simple language is easy to learn.

### Functions (and flow control and variables)

CMake's scripting language is as simple as it gets. There are variables, functions, and a few
flow-control constructs.

The interesting thing is that all these manifest as functions. Both "declaring" variables as well as
assigning values is done through functions. There is no real variable declaration other than
creating an empty varible with the given name.

Flow control also manifests through function calls, at least syntactically. Ultimately, CMake
scripts look like a series of function calls with nothing in between.

A function call (as one might guess from the previous code snippet) looks like the following:

```CMake
function(ARG1 ARG2...)
```

where `function` is the name of the function, and `ARG1` and `ARG2` are arguments to it. (The `...`
only indicates that a function may have an arbitrary number of arguments.) Note that CMake is
case-insensitive regarding function names, so

```CMake
FUNCTION(ARG1 ARG2...)
```

works equally well, just as

```CMake
function (ARG1 ARG2...)
```

with arbitrary white-space between function names, their arguments, and in between function
invocations. The same cannot be said for arguments, as there is no delimiting character between
arguments. _(See [Types](#types) below.)_

One will encounter just about all of the above styles in the wild. Use a style that is most readable
to you.

_NOTE: advanced CMake users might have noticed I have abused the `function` "keyword" which is
actually a built-in function name used to define user-defined functions. This crime was done solely
for the sake of syntax highlighting in Markdown. User-defined functions will come much later in the
tutorial, as one can get 95% of the way without custom functions._

### Types

Perhaps the most troubling part of the language is its type system, or to be precise, the lack of
it. The only type is the _string_ type. __Everything__ is a string in CMake! With this clear, many
of the peculiarities will become apparent.

Because CMake has been around for some time, and the scripting language evolved over time to be more
user-friendly, one may encounter coding styles that are different from the one presented here. These
differences serve as good examples to understand that everything is a string, as well as the way
variables behave.

### Variables

CMake is a Turing-complete scripting language (no proof, but seems much like it). This does not mean
one should do more with it than getting a build system running. A crucial part of an imperative
scripting language is the notion of variables.

One creates a variable in the following way:

```CMake
set(MYVAR "Value of my variable")
```

The `set` function is used to create new variables, set the contents of existing ones, as well as
"declare" a variable. Quotation marks are needed because there is not much difference in having an
empty variable as opposed to not having one at all. _(The difference is detectable, but essentially
is able to code only a single bit of information.)_

We have already promised (but not yet proven that)

```CMake
message("Hello, CMake!")
```

instructs CMake to emit messages to the console at the time the command is encountered. The argument
to `message` is a string. Let's take this argument from a variable.

```CMake
set(GREETING "Hello, CMake!")

message(GREETING)
```

What would we see if we ran this CMake script?

```
GREETING
-- Configuring done
-- Generating done
-- Build files have been written to: ...
```

Why don't we see the message we actually wanted to? Starting from CMake 3.0 (which means most
scripts since 2015), one is allowed to omit the quotation marks to denote a string. Therefore

```CMake
message(GREETING)
```

does nothing more than provide an all-capitalized string as the argument to `message`. If we
actually wanted to get the _contents_ of our variable named `GREETING`, we would have to
"dereference" its name. Users of Bash will find dereferencing familiar. The script

```CMake
set(GREETING "Hello, CMake!")

message(${GREETING})
```

will output the expected

```
Hello, CMake!
-- Configuring done
-- Generating done
-- Build files have been written to: ...
```

The ability to omit quotation marks around strings has some interesting consequences.

```CMake
set(VERB Hello)
set(NOUN CMake)

set(GREETING "${VERB}, ${NOUN}!")

message(${GREETING})
```

This example, apart from showing that dereferencing is "stronger" than quotation marks, actually
outputs what we'd expect. What happens if we omit the quotation marks, which we've learned we are
allowed to do?

```CMake
set(VERB Hello)
set(NOUN CMake)

set(GREETING ${VERB}, ${NOUN}!)

message(${GREETING})
```

outputs

```
Hello,CMake!
-- Configuring done
-- Generating done
-- Build files have been written to: ...
```

One space got left behind. If we were to take a look at the documentation page of `message`, we'd
see something like this:

```CMake
message("message to display" ...)
```

The `message` command takes an arbitrary number of string to display, which it will concatenate
without any joining characters. Without placing quotation marks around `${VERB}, ${NOUN}!`, the two
became separate function arguments. The only time one needs quotation marks, is if he/she wishes to
guard spaces from becoming white-spaces.

If we wanted to concatenate two strings (fairly common when assembling file paths for eg.) we could
do so with

```CMake
set(BASEPATH ./)
set(MAIN "${BASEPATH}main.cpp")
```

or

```CMake
set(BASEPATH ./)
set(MAIN ${BASEPATH}main.cpp)
```

If there are no spaces to guard, we are safe to omit the quotation marks.

_NOTE: we will never refer to the working directory as such, there will be special variables that
allow us to refrain from error-prone solutions, such as the above._

## Invoking CMake

Up until this point, we have silently omited the actual invocation of CMake. To allow the reader to
follow along and verify all that is shown here, one needs to put to use hopefully some skills
already possessed.

### Pre-requisites

Because CMake generates makefiles of a given flavor, we will very soon need an actual makefile
execution engine installed.

#### Linux

On Linux, luckily the default generator is `Unix Makefiles` which is most likely already installed.
If not, one can install it saying

```bash
sudo apt install make
```

or the equivalent command on your distro.

#### Windows

On Windows, there are gazillions of options available to start. For the sake of this tutorial, we
will take an incremental approach and favor *nix-style, command-line usage. Also, to obtain output
identical to `Unix Makefiles`, we will use the somewhat outdated NMake tool.

To install the NMake build system, we will hit two birds with one stone, since it is bundled with
the Microsoft Visual C++ compiler (which we will need later anyway).

Obtain the Build Tools for Visual Studio 2017 from
[here](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017), and install only
the Visual C++ compilers and libraries as well as the build tools. This will install both MSBuild
and NMake.

Because on Windows, the installer will not place these tools onto the `PATH` (for good reasons), one
can only use both the compiler and build tools from a so called `Developer Command Prompt`. This is
an ordinary `cmd.exe` after invoking `vcvarsall.bat`, a batch script that is bundled with the
compiler. This script sets up extra environmental variables required for the build tools and the
Visual C++ compiler to work. In the Start Menu, under All Programs, find the folder named Visual
Studio 2017, and launch the `x64 Native Tools Command Prompt for VS 2017` (assuming you are running
an x64 machine).

### Out-of-source invocation

It is good practice to build compiled languages out-of-source, meaning that all byproducts of the
compilation process do not mingle with the actual source files. One can achieve this by building in
a subfolder of the sources, or in a totally unrelated directory.

Let's say we have a folder structure that looks like the following:

```
my_project -+
            |
            inc -+
            |    ...
            |
            src -+
            |    ...
            |
            CMakeLists.txt
            |
            README.md
```

In this case, we could either create a `build` folder inside `my_project`, or we could create a
`build` folder in a totally different location.

Before we actually invoke CMake, there is one more useful command to be familiar with to speed up
the process. While our original

```CMake
message("Hello, CMake!")
```

script would do just as we intended it to, it might be useful to add just one more line to it.

```CMake
project(Hello LANGUAGES NONE)

message("Hello, CMake!")
```

By default, CMake assumes one is using it to compile C/C++ applications, and as such it tries to
look for a C/C++ compiler installed on the machine. Even if one has a properly configured compiler
and command-line, this detection takes time, which for the moment is totally unneccessary.

The `project` function takes a project name, after which one may specify the languages that the
project will use. As a side-effect, it will create variables we might use later on.

As for the actual invocation, let's say we created a `build` subdirectory. Inside that directory, we
might give the following command:

| Linux | Windows |
| ----- | ------- |
| `cmake ../` | `cmake -G "NMake Makefiles" ..\` |

which will instruct cmake to look for a `CMakeLists.txt` file one folder up from the working
directory. The path provided may be relative or absolute. Makefiles will be generated in the working
directory (the place from where we invoked CMake).

Unfortunately, there is no way to tell CMake which is the default generator we would like to use
system wide. The default on Windows is `Visual Studio 15 2017`; the generated MSBuild files are
fairly verbose on the command-line and not ideal for learning.

_If the extra typing troubles Windows users, bear with me, because ultimately we will leave behind
the command-line as it is and use CMake from an IDE that supports it. CLI usage is solely for the
sake of better understanding what's going on under the hood._

### Result

These are the results on the various platforms.

#### Ubuntu 16.04

```
$ cmake ../
Hello, CMake!
-- Configuring done
-- Generating done
-- Build files have been written to: ...
$ cmake --build .
$
```

#### Windows 10

```
$ cmake -G "NMake Makefiles" ..\
Hello, CMake!
-- Configuring done
-- Generating done
-- Build files have been written to: ...
$ cmake --build .
$
```

The second `cmake --build .` invocation is asking cmake to call on the generated build system to
execute the makefiles.

Notice how nothing happens once we execute the generated makefiles. This is to be expected, because
there was no actual work placed in the CMake scripts. The `message` command only has meaning at
configuration-time, hence the message we see __before__ we see the "Configuration done" message.

## Debugging

Even though the reader has been warned not to do too much in CMake other than getting a build going,
experienced users will venture into doing fairly complex things. In such cases, and when one is
still learning CMake, it is bound to happen that things go wrong within the CMake scripts. There are
three tools available for debugging.

### message()

The `message` command is the CMake equivalent of the C-runtime `printf` function. It is useful when
one suspects that the value of a variable is not quite what one wished it to be.

### Documentation

Speaking of `message`, this is the opportune moment to introduce the user to the
[cmake documentation](https://cmake.org/documentation/) which is a most useful resource for finding the
capabilities of all commands available, as well as a [list of useful
variables](https://cmake.org/Wiki/CMake_Useful_Variables). Elements of both resources will gradually
be introduced throughout this tutorial.

Taking a final, closer look at the docs of the [message
command](https://cmake.org/cmake/help/latest/command/message.html?highlight=message), we will see
that the actual definition is:

```CMake
message([<mode>] "message to display" ...)
```

Besides accepting an arbitrary number of strings to display, there is also an optional first
argument to indicate the seriousness of the message. The docs are self-explanatory, no further
comments needed beside: __this is the single most useful command when learning CMake__.

### Verbose makefiles

CMake may configure without errors, and the values of all variables may seem fine, yet still fail
compilation with compiler or linker errors. In this case, it might be useful to check the actual
tasks executed. __The proper way__ to do so __is not by looking at the generated makefiles__.

Verbose makefiles print out every command executed by the build system. The means of obtaining a
verbose invocation depends on the target generator.

All such commands make use of the ability to pass parameters native to the build system through the
cmake build driver. All arguments after the final double dash `--` will be passed onto the target
build system.

#### Ninja, Unix & NMake Makefiles

```
cmake --build . -- VERBOSE=1
```

#### MSBuild (Visual Studio generators)

```
cmake --build . -- /v:diag
```


<br><br>

----------------------------------------------------------------------------------------------------
⏫ [Back to Top](../)
        [Next Lesson](../Lesson1_CompileC_CPP/) ⏩
