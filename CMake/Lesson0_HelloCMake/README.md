# Lesson 0 - Hello CMake


As in all scripting and programming tutorials, we will begin with a simple, "Hello CMake" script. The script looks like the following:

```CMake
message("Hello, CMake!")
```

Before we could go into detail what's going on in this single line of code, we must first talk a little about how CMake works. _Much like one would have to give a jump-start to the C/C++ compilation model when confronted with the first "Hello World" in either of the two._

## CMake internals

CMake essentially gets its job done, creating makefiles in the falvor of choice by going through two steps in order: __configuration__ and __generation__.

### Configure

The configure step is the first. This is when CMake parses and processes the CMake scripts the user provides. Errors in the scripts, may them be syntactic or logical, will produce configuration-time errors which more often than not will halt the processing of the CMake scripts, aborting the entire process. Some logical errors CMake cannot forsee (much like run-time errors in all programming languages) will create errors during the build process.

### Generation

Once the scripts have been processed and no errors have been encountered, CMake enters the makefile generation phase. This is the time when the set of processed CMake commands are turned into makefiles which are written to disk.

## CMake scripting language

Perhaps the most taunting part of CMake is its scripting language. Compared to modern script languages, CMake has a fairly simple, unexpressive scripting language. The reason for its simplicity, is that it is the common denominator of all build systems. It does not wish to surface features which are complicated (or impossible) to implement in a back-end.

### Functions (and flow control and varibles)

CMakes scripting language is as simple as it gets. There are variables, functions and a few flow control constructs.

The interesting thing about these is that all these manifest as functions. Both "declaring" a variables as well as assigning a value to it is done through a function. There is no real variable declaration, other than creating an empty varible with the given name.

Flow control also manifests through function calls, at least syntactically. Ultimately CMake scripts look like a series of function calls with nothing in between.

A function call (as one might guess from the previous code snippet) looks like the following:

```CMake
function(ARG1 ARG2...)
```

where `function` is the name of the funciton, and `ARG1` and `ARG2` are arguments to it. _(... only indicates that a function may have an arbitrary number of arguments)_ Note, that CMake is case-insensitive regarding function names, so

```CMake
FUNCTION(ARG1 ARG2...)
```

works equally well, just as

```CMake
function (ARG1 ARG2...)
```

denoting white-space insensitivity between function names and their arguments and in between function invocations. The same cannot be said for arguments, as there is no delimiting character between arguments. _(see later Types.)_

One will encounter just about all of the above styles in the wild.

### Types

Perhaps the most troubling part of the language is its type system, or to be precise, the lack of it. The only type is the _string_ type. __Everything__ is a string in CMake.

The message command instructs CMake to emit messages to the console at the time the command is encountered.