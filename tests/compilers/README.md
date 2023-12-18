# Compilation tests

This directory contains one file, `gpu_checks`. This file contains tests related to the compilation and performance of two basic GPU tests, `warmup` and `muiltigpu`. The two tests in question are part of the `performance_modelling_tools` repository, which is included in the `common` directory in the root directory of this repo, which a sybmolic link to is found in the `src` directory.. The tests offer various compilation options (e.g. HIP, CUDA, OMP, ACC, HIP/CUDA + OMP/ACC) via Makefiles, which will need to be edited accordingly for your setup. These Makefiles are located in the `src` directory.

The tests are split into compile-only and run-only tests, the latter of which depend on the former (i.e. the run-only test will only execute if the corresponding compile-only test passes). This design allows a test failure to be instantly diagnosed as being a compilation issue or performance/runtime issue. All run-only tests are performance tests, and so it is recommended to alwasy run these tests with the `--performance-report` command-line option passed to ReFrame.

These tests are intended as a way to investigate the differences between various compilation options, both at compile-time (i.e. can this code be compiled successfully this way on my system) and run-time (how do different compilation methods and flags, etc. affect performance of the code at runtime). 

Make sure to check `setup_files/test_env.config` before running these tests (details of this file in the main repo README). All tests will include all `gpu-env`, `gpu-mod`, and `gpu-cmd` lines.

Below is a brief outline of each test. This is to inform users of what the motivation behind the test, what it does, and what, if any, parts of the test users might want to modify for their own purposes.

## `gpu_checks.py`

### `gpu_compile_base_check`

This is not a test in and of itself, but instead a base check for the two compilation tests. It simply sets the system and programming environment the test is to run on (which needs to be set before the test will run successfully) and the build system.

### `warmup_compile_check` and `multigpu_compile _check`

These two tests compile the code for the two corresponding run-only tests using the associated Makefiles. They set the compilation environment and have one parameter, `build_type`, which sets how the code will be compiled. This can be modified as desired. The test passes if the compiled executable exists

### `warmup_check` and `multigpu_check`

These two tests run the compiled code generated from the corresponding compile-only tests. Both tests have the same two test parameters, `ngpus_per_node` and `build_type`, which both can be modified as desired. Note that the `build_type` parameter will need to match the same parameter in the corresponding compilation test, otherwise the dependencies between the tests will not work correctly and certain test variants will not run at all. The code following `THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET` will need to be set before running these tests. Job options can also be modified as desired, whetehr in the class initialiser or in the `set_job_opts()` method.

Both of these tests can measure performance if `--performance-report` is passed to ReFrame. The parts of each test pertaining to performance will likely need to be modified. These are the methods `set_perf_dict()` and `extract_timing()`, the latter of which is marked with the `@performance_function` decorator. The reference values, defaults of which are set immediately after `PERFORMANCE OF DIFFERENT SECTOINS OF CODE` will also likely need to be altered
