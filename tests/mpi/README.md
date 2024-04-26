# MPI tests

This directory contains two test files, `mpi_checks.py` and `gpu_mpi_checks.py`. As the names suggest, the former contains all MPI tests which are designed to run on CPU nodes, while the latter are tests designed to run on GPU nodes and test GPU-MPI communcation. There are two corresponding `.yaml` configuration files which define system-level parameters, environment configurations, job options, and test parameters for the tests. The `src` directory holds all source code files and Makefiles needed for compiling and running the code used in the tests. The `common` directory is a symbolic link to `common` in the root directory of this repo. This contains the `profile_util` library and node health checker script, which are both used extensively in these tests.

Below is a brief outline of each test in the two files. This is to inform users of what the motivation behind the test, what it does, and what, if any, parts of the test users might want to modify for their own purposes.

## `mpi_checks.py`

### `MPI_Comms_Base`

This is not a test in and of itself, but a base test class which other tests inherit from. It sets up the test environment, builds the `profile_util` library, sets compilation flags, job options, sets up the node health check, and defines the basic sanity check for many of the tests.

### `Pt2Pt`

This test is a performance test for MPI point-to-point communication. The program executed takes several arguments to control the amount of data each process generates, the scope of communication (all processes communicate with all other processes, or only adjacent processes), the type of communication (from fully blocking to fully asynchronous), and more. The test has two parameters, `num_nodes` and `num_tasks_per_node`. This test also records performance when `--performance-report` option is passed to ReFrame at runtime.

### `CollectiveComms`

Essentially the same test as `Pt2Pt` except the program performs various MPI collective communication routines instead of point-to-point communication. Again, performance reference values, test parameters, and executable options can be modified as desired.

### `DelayHang`

We experienced an issue where if there was a large enough delay between when a process performs a send and the receiving process posts the corresponding receive that codes could hang. This was seen in a pipeline of a radio astronomy team using our system. This test was designed to track this issue. A subsequent upgrade of our libfabric library resolved the issue, but we have kept the test in case other unexpected behaviour in this scenario arises again.

### `CorrectSends`

This test does not inherit from `MPI_Comms_Base`. This test was developed when we noticed that data could be corrupted when using `MPI_Isend` - the data sent by one process did not match the data received by the receiving process. This data corruption did not happen in every instance, so the test runs the program many times (default 50, can be modified if desired). Again, an upgrade to our libfabric library resolved the issue, but we have kept the test. The test has one parameter, `send_mode`, which runs the program with different version of the MPI send routine.

### `LargeCommHang`

This test is designed around an issue that we have observed on Setonix for at least a year and is still present. MPI codes using a large enough `MPI_COMM_WORLD` with a large enough number of processes and each process sending and receiving a large enough amount of data can hang. This test has one test parameter `ntasks_per_node`.

### `LargeCommLibfabric`

This test is designed around an issue that we have observed on Setonix for at least a year and is still present. MPI codes using a large enough `MPI_COMM_WORLD` with a large enough number of processes and each process sending and receiving a large enough amount of data can sometimes crash while displaying libfabric errors. Unlike the similar `LargeCommHang` issue, this is not 100% reproducible. This test has one test parameter `ntasks_per_node`.

### `MemoryLeak`

This test does not inherit from `MPI_Comms_Base`. This test was motivated by several observed instances of programs failing with "out of memory" errors when it seemed like there should be memory available. Subsequent investigation showed that there was a memory "leak" of sorts when running MPI codes - memory being used by the code would not be released properly at certain points, leading to a build-up of consumed memory on the node(s) on which the program was running. This meant that over time, as more MPI codes were run on a node, the free memory on said nodes (even when nothing was being run) would decrease over time. The only way to fix the affected nodes was to reboot them. An upgrade to our libfabric library removed this persistent memory accumulation on a node.

This test, however, still detects temporary "memory leaks" when running MPI programs across multiple nodes. The memory no longer accumulates after the program has finished running, but during execution there can be points where the memory reported being used by the program does not match what is being consumed on the node. The exact discrepancy depends on the scale of the program being run. This test runs two programs simultaneously, the main MPI program and one which monitors the memory usage of every process being used by the main program and reports that memory at a constant cadence. This allows monitoring of memory usage while the main program is within an MPI routine itself. The memory used by the processes is compared to the total consumed memory of the node, and if there is a large enough discrepancy (default 10%) at any point the test fails.

Executable options to control the amount of data, job options, test parameters (`num_nodes`, and `ntasks_per_node`), and cadence of memory reporting can all be edited as the user desires. Importantly, the setting `exclusive: True` should not be altered. If other programs are allowed to run on the node, then the utility of comparing consumed node memory with memory reported by the MPI program is useless.

## `gpu_mpi_checks.py`

### `gpu_mpi_comms_base_check`

This is not a test in and of itself, but a base test class which other tests inherit from. It sets up the test environment, builds the `profile_util` library, sets compilation flags, job options, sets up the node health check, and the test parameters for all child tests, `num_nodes` and `ngpus_per_node`. All other tests in this file inherit from this base check.

### `gpu_copy`, `gpu_sendrecv_check`. `gpu_allreduce_check`, `gpu_async_sendrecv_check`

These tests check the performance of MPI GPU -> GPU copy, performance (bandwidth) test for GPU-MPI single send/recv operations, performance of GPU-MPI all-reduce operation, and performance of GPU-MPI asynchronous send/recv operations, respectively. The only thing that should need to be edited are the performance reference values.

### `gpu_correct_sendrecv_check`

This test is analogous to `CorrectSends` from `mpi_checks.py`. This is to test that the data sent by the sending process matches the data received by the corresponding receiving process when performing GPU-MPI single send/recv operations. There is no performance element to this test, and therefore should not need to be edited in any way.
