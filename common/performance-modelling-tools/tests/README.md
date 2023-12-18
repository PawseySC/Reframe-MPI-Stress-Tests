# Tests
Collection of simple tests for GPUs. These tests fall under the categories of `warmup` and `multigpu` tests. All tests are compiled via a Makefile and support several compilation options including CUDA, HIP, OpenMP and OpenACC. All tests generate reports at runtime detailing level of parallelism, device information, and performance.

More detailed information about the tests can be found in the corresponding directories.

## warmup

This test checks to see how a GPU warms-up. It runs a few simple instructions for a set number of rounds (the warmup) and then runs a larger set of device instructions labelled as the run kernel for a set number of iterations. The instructions are basic, standard operations such launching a kernel, moving data from device to host and vise versa, and allocating memory.

## multigpu

This test runs a set number of  rounds of kernel instructions concurrently on all available gpus. These instructions are basic, standard operations such as allocate memory, transfer to and from host and device, computation on device, and memory deallocation. Options are available to adjust how the operations are performed.

## profile_util

This is not a test, but rather a C++ library that provides inline logging of memory, timing, and affinity within code. The library can be built through one of several scripts or through CMake. Each script builds different versions of the library. The build scripts currently include support for serial, OpenMP, MPI, OpenMP + MPI, CUDA/HIP, CUDA/HIP + OpenMP, CUDA/HIP + MPI, and CUDA/HIP + MPI + OpenMP.
