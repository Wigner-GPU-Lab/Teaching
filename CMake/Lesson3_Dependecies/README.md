# Lesson 3 - Dependencies


One of the strengths of CMake is its ability to find dependencies with minimal user effort (based on upstream support). There are two catgories of dependecies, both with their preferred way of detection:

- Upstream is a binary or is built using another technology
- Upstream was built using CMake

There is also a hidden third option:

- Upstream is built using another technology but ships with CMake support

## Find Module dependency detection

Find module scripts target upstreams that are not CMake aware, meaning that they either ship as a binary or are built using other build systems. The layout of such libraries/tools are too numerous to count, not to mention their install locations may be unique, or worse: user defined. Guessing user defined install locations in advance requires an IT oracle. Without otherwordly support, depending on such libraries will need minimal user interaction.

The scripts that detect such dependencies are called Find module scripts and are named by convention as `FindName.cmake`, where "Name" is the name of the dependency, for eg. `FindMPI.cmake`. Authoring such scripts is outside the scope of this lesson, but will be explained in a later lesson. For the time being, we'll restrit ourselves to browsing the comments section of these scripts.

CMake (3.9) comes with 148 pre-installed find module scripts. _(On Windows they are located under `<install root>\share\cmake-3.9\Modules` and on Ubuntu under `/etc/share/usr/share/cmake-3.5/Modules/`)_ You might want to scan through the list to get a feeling of what comes bundled with CMake. Let's take a look at FindMPI.cmake, shall we?

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

Up until this point, the comments section demonstrates, how to depend on MPI using the "old school" way. The comments are fairly straightforward. Once the scripts are run, they set certain variables which we may use in our build scripts.

How do we run these find module scripts? We use the `find_package` command likt this:

```CMake
find_package(MPI REQUIRED)

add_executable(app Main.c)

target_include_directories(app PRIVATE ${MPI_C_INCLUDE_PATH})

target_link_libraries(app PRIVATE ${MPI_C_LIBRARIES})

set_target_properties(app PROPERTIES LINK_FLAGS ${MPI_C_LINK_FLAGS}
                                     COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
```

If you think to yourself, this seems like a great deal of work only to link to MPI, you are right. (Hold on a sec, we'll have you covered.) All this is required because there are quite a few MPI implementations out in the wild, and `FindMPI.cmake` knows how to use just about all of them. Some require just include directories set up, some require only linking, others require extra compile definitions... even you you know for a fact, that the MPI flavor you have installed only requires linkage per say (in which case variables like `${MPI_C_COMPILE_FLAGS}` are empty), others have gone to great lengths in order to collect all the MPI variations and the means of consuming them. Sometimes it's nice to just not think about stuff and do as one's told and stay rest assured that it's always going to be fine.

If you (dear reader) were one of the few who thought, that is a lot of typing, let's read on the comments section:

```rst
# Additionally, the following :prop_tgt:`IMPORTED` targets are defined:
#
# ``MPI::MPI_<lang>``
#   Target for using MPI from ``<lang>``.
```

An IMPORTED target is a library that is not meant to be built, but only collects certain properties to be inherited by other targets. This comes in handy right now (and was created for just this case), because the MPI library is already built, we have no work with it; we would only like to inherit a set of compiler and linker options.

So how do we use an imported target?

```CMake
find_package(MPI REQUIRED)

add_executable(app Main.c)

target_link_libraries(app PRIVATE MPI::MPI_C)
```

Wow! That is much simpler. All the properties we set manually earlier are now all inherited from the imported target. The first part of imported target, before `::` is always the name of the module at hand. The second part are "sub-modules", parts of the module than can stand by themselves. In the case of MPI, an implementation may omit fortran bindings for eg., but that doesn't mean we couldn't rely on the C bindings.

Following parts of the comments section of `FindMPI.cmake` are useful for building custom build targets. These will be explained in a coming lesson of this tutorial.