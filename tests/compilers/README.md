# Compilation tests

This directory contains three files, `gpu_checks.py`, `assembly_checks.py` and `optimise_checks.py`. 

The first file contains tests related to the compilation and performance of two basic GPU tests, `warmup` and `muiltigpu`. The two tests in question are part of the `performance_modelling_tools` repository, which is included in the `common` directory in the root directory of this repo, which a sybmolic link to is found in the `src` directory.. The tests offer various compilation options (e.g. HIP, CUDA, OMP, ACC, HIP/CUDA + OMP/ACC) via Makefiles, which will need to be edited accordingly for your setup. These Makefiles are located in the `src` directory.

The tests in `gpu_checks.py` are split into compile-only and run-only tests, the latter of which depend on the former (i.e. the run-only test will only execute if the corresponding compile-only test passes). This design allows a test failure to be instantly diagnosed as being a compilation issue or performance/runtime issue. All run-only tests record performance of various parts of the GPU codes, such as h2d and d2h speeds.

`assembly_checks.py` tests the affect of different compilers and/or compilation flags on the assembly code produced by the compiler. It counts the occurence of a set of 1 or more instructions (depending on the value of the test parameter in the configuration file) in the assembly file. Test parameters include compiler flags so that different options can be compared to each other. Users can simply add any desired flags they want to explore as a test parameter to the configuration file.

`optimise_checks.py` is very similar to `assmembly_checks.py`. Instead of checking the presence of assembly code instructions it checks for optimisations, such as vectorisation and loop unrolling, present in the compiler-generated optimisation file. There is also a performance test to investigate the correlation between certain optimisations and runtime performance.

These tests are intended as a way to investigate the differences between various compilation options, both at compile-time (i.e. can this code be compiled successfully this way on my system) and run-time (how do different compilation methods and flags, etc. affect performance of the code at runtime). 

Below is a brief outline of each test. This is to inform users of what the motivation behind the test, what it does, and what, if any, parts of the test users might want to modify for their own purposes.

## `assembly_checks.py`

### `countInstructions`

This test compiles a simple piece of code containing some basic vector maths. It checks the generated assembly file for the presence of one or more assembly code instructions (e.g. `fma`, which is the default instruction). There are several test parameters, all of which are compilation flags except for one defining what instruction(s) to search for. The test passes if there is at least one occurence of the instruction in the assembly file. Users are free to add other compiler flags as test parameters or change the source code file to be compiled.

## `optimise_checks.py`

### `countOptimsations`

This test is analogous to `countInstructions` detailed above. It is the same except that it counts the number of optimisations in a compiler-generarted optimisation file rather than assembly instructions in an assembly code file. The default optimisations checked are vectorisation and loop unrolling, defined in a test parameter. Other test parameters are compiler flags.

### `benchmarkOptimisations`

This test compiles and runs a simple piece of code containing some basic vector operations - allocating memory, initialising, and doing some vector maths. It takes the same test parameters as `countOptimisations`. Rather than just checking the number of optimisations in the optimisation file, however, it also checks for performance of each stage of the associated code at runtime. In this way, the affect of copmiler flags and the presence of certain optimisations on code performance can be seen.

## `gpu_checks.py`

### `gpu_compile_base_check`

This is not a test in and of itself, but instead a base check for the two compilation tests. It simply sets the system and programming environment the test is to run on and the build system.

### `warmup_compile_check` and `multigpu_compile _check`

These two tests compile the code for the two corresponding run-only tests using the associated Makefiles. They set the compilation environment and have one parameter, `build_type`, which sets how the code will be compiled. This can be modified as desired. The test passes if the compiled executable exists.

### `warmup_check` and `multigpu_check`

These two tests run the compiled code generated from the corresponding compile-only tests. Both tests have the same two test parameters, `ngpus_per_node` and `build_type`, which both can be modified as desired. Note that the `build_type` parameter will need to match the same parameter in the corresponding compilation test, otherwise the dependencies between the tests will not work correctly and certain test variants will not run at all. Job options can also be modified as desired. Both of these tests measure performance. The reference values defined in the configuration file will likley need to be modified.
