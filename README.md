# Reframe-MPI-Stress-Tests
A collection of tests to be run using Reframe.

## Types of tests

The tests fall under three broad categories. A brief overview is given here, but more detail about the individual tests is provided in the README in the corresponding directories.:

### MPI

A collection of MPI tests. These tests cover various types of MPI communication (point-to-point and collective) in various workflows designed to more closely replicate typical users workflows seen on our HPC system. They are also written in such a way that they can serve as stress tests for the system, able to be run at large scale with many processes across many nodes, which is often atypical of standard MPI tests run on HPC systems (e.g. OSU microbenchmarks). The tests cover usage of MPI in CPU-only code as well as GPU-enabled code. All MPI tests also include a node health check at the beginning and end of each run and logging of environment information through the settiong of certain environment variables.

### SLURM

A collection of SLURM tests. These tests cover particular issues, problems, and strange behaviour we have seen with our SLURM installation(s), so may not be as widely applicable as the MPI and compilation/performance tests. Nevertheless, they serve as an example of the type of tests which could be used to monitor certain aspects of a centre's SLURM installation. These tests cover such aspects of resource allocation vs. request (e.g. ensuring what a user asks for in terms of accessible memory, cores, etc. is what is actually provided to them), account billing, affinity of OMP threads and MPI processes, and node configuration ith respect to SLURM config.

The CPU and GPU nodes on Setonix have different hardware and setups, and as such the SLURM setup on the two sets of nodes is not the same. Therefore, as with the MPI tests, we have tests for SLURM use on both CPU and GPU nodes.

### Compilation/Performance

A collection of tests which focussed on compilers and the affect of different compile-time options, flags, etc. on the performance of the compiled code. The tests are split into compilation tests and performance tests, the latter of which depend on the former. Therefore, if the test fails, you will instantly know if it's a compilation failure or code execution/performance failure. By running the tests with multiple different combinations of compilers, flags, etc. one can investigate the effects with these tests.

## How to run tests

To run the tests one only needs to install ReFrame on their system and modify the `settings.py` and `test_env.config` files for their system. Slight modifications to the test files will also be needed for the `valid_systems` and `valid_prog_environs` ReFrame variables. All tests are able to run as is on Setonix, but depending on your setup, you might also need to adjust test parameters (for tests which have them) and job scheduler options. Once all that is done, one simply needs to execute the following command to run the tests.

`reframe -C ${PATH_TO_REPO_ROOT_DIR}/setup_files/settings.py -c $PATH_TO_TEST_FILE_OR_DIRECTORY -r [-t $TAG ...]`

where `...` stands for any other optional ReFrame command-line araguments one wants to pass.

### Configuring the tests for your system

The tests have been written in a generalised way to try and make them as portable as possible. Nevertheless, there are some steps you will have to make before you are able to run the tests on your system. You need to add your system(s) into the `setup_files/settings.py` file. We have a placeholder generic system that users can edit as needed. 

Also included is a configuration file, `setup_files/test_env.config`. This file stores environment variables, modules, and commands which are included in relevant tests. For example, all lines in the config file starting with `mpi` (`mpi-env`, `mpi-mod`, `mpi-cmd`) will, in all MPI tests, have those environment variables set, modules loaded, and commands run. The script `common/scripts/set_test_env.py` parses this file, and the `set_env()` method of that script is invoked in test body definitions. Setonix uses a CRAY-MPICH MPI implementation and lmod module management system, and sensible options for our system are set there. It can be edited as needed to suit your setup.

Finally, within the test definitions themselves, there are sections of code preceded by 

#########################
#                       #
#########################

These mark sections of the test which the user might need or want to edit. Typically these mark test parameters, job scheduler options, and system-specific variables. 


