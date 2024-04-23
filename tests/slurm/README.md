# SLURM tests

This directory contains two files, `slurm_cpu_checks.py` and `slurm_gpu_checks.py`. As the names suggest, the former contains all MPI tests which are designed to run on CPU nodes, while the latter are tests designed to run on GPU nodes. On Setonix, the GPU and CPU nodes have different SLURM setups to accommodate the different hardware/specs of the nodes.. There are two corresponding `.yaml` configuration files which define system-level parameters, environment configurations, job options, and test parameters for the tests. The `src` directory holds all source code files and Makefiles needed for compiling and running the code used in the tests. The `common` directory is a symbolic link to `common` in the root directory of this repo. This contains the `profile_util` library that is used extensively in these tests.

These tests check SLURM functionality and behaviour. These tests might have less utility on other systems since some of them were motivated by very specific issues we were observing or by our particular SLURM configuration and setup. Nevertheless, we feel they can serve as examples of the types of SLURM checks that have proved useful for us to have, and a sufficient base for users to perhaps modify to suit their particular SLURM configuration and setup. Broadly, the tests check correct resource allocation and account billing given a certain allocation request, check for optimal affinity, and functionality of different types of workflows (e.g. regular job, job arrays, job packing, heterogeneous jobs)

Below is a brief outline of each test in the two files. This is to inform users of what the motivation behind the test, what it does, and what, if any, parts of the test users might want to modify for their own purposes.

## `slurm_cpu_checks.py`

### `slurm_node_mem_check`, `slurm_cpu_mem_check`, and `MemoryCompileTest`

These three tests check that when a SLURM resource allocation requests includes an explicit memory request, other than `--exclusive` or `--mem=0`, that they actually have access to the requested amount of memory in the job. `MemoryCompileTest` compiles the memory reporting code and passes if the executable is compiled successfully, whereas `slurm_node_mem_check` and `slurm_cpu_mem_check` test per-node memory requests (via `--mem`) or per-CPU memory request (via `--mem-per-cpu`). Both tests attempt an srun call with slightly below the memory asked for and one with slightly more, and checks that they run/are rejected appropriately. The tests have one parameter, `multithreading`, which sets whether the job is run with 1 or 2 threads per core. All job options in the test can be modified as desired.

### `billing_check`

This test compares the billing determined by SLURM (from the `billing`, `NumCPUs`, `mem`, and `TRES=cpu` fields from `scontrol show job $SLURM_JOBID`) to what should be billed given the resource allocation request. This test has two test parameters - `multithreading` and  `ncpus_per_task`. They are set to cover several different scenarios, mainly 1 or 2 threads per core and whether the requested memory is more or less than the equivalent memory of the requested number of CPUs.

Note - for this test to work, the `processor` dictionary needs to be set in the `setup_files/settings.py` settings file for each system + partition combination on which you want to run the test.

### `slurm_cpu_check`

This test is similar to the memory tests described above. It checks that given a certain resource allocation request the granted job has access to the correct number of cores. This test has one parameter, `multithreading` to set the number of threads per core. The job options specified can be modified as desired.

Note - for this test to work, the `processor` dictionary needs to be set in the `setup_files/settings.py` settings file for each system + partition combination on which you want to run the test.

### `omp_thread_check`

This test checks that the number of OMP threads reported by the OMP API matches what should be available given a certain resource allocation request. We found that the presence (or lack thereof) of `--cpu-bind` option in the srun call (and the value it takes) can affect the number of threads reported by the OMP API. This discrepancy is still occurring on Setonix, and the default values of the job options and test parameters (`multithreading` and `cpu_binding`) do showcase this issue.

### `affinity_check` and `AffinityCompileTest`

This test checks the automatic affinity calculations for OMP threads and MPI processes. `AffinityCompileTest` compiles the affinity reporting program, and `affinity_check` runs the program, parses the output using a separate python script, and checks the logic to see if the affinity of the OMP threads and MPI processes (separately) is optimal. There are many test parameters for `affinity_check` which cover different OMP placement options, hyperthreading, and some job options. The default test parameter values and job options cover an extensive set of combinations, which can be modified as desired.

Note - for this test to work, the `processor` dictionary needs to be set in the `setup_files/settings.py` settings file for each system + partition combination on which you want to run the test.

### `het_job_test`

This test is designed to check that various workflows for heterogeneous job submission work as intended. Users from one of our research groups had trouble launching heterogeneous jobs, which motivated the development of this test. It tests three different modes of submitting heterogeneous jobs in SLURM. The original issue was fixed with a SLURM patch, but have kept this test as a way of monitoring the functionality of heterogeneous job support in our SLURM installation.

The test parameter for this test controls the mode of heterogeneous job submission. The job options and executable options, set in the configuration file, can be modified as needed.

### `accounting_check`

This test checks that SLURM accounting is being calculated corectly. It runs a simple job which simply sleeps for a certain amount of time and then check that the accounting from `sacct` matches what it should given the job's resource allocation request.

## `slurm_gpu_checks.py`

### `gpu_count_check`

This test checks that the correct amount of GPUs are available to a job given that job's resource allocation request. This test queries `rocm-smi` to check the number of GPUs available given we have AMD GPUs on Setonix. The test parameter controls the type of job (exclusive or shared).

### `gpu_cpu_count_check`

This test checks that the correct amount of CPUs are available to a job given that job's resource allocation request. On Setonix, each GPU grants a certain number of CPUs, so if the right number of GPUs are granted, the right number of CPUs should be granted too. However, we have observed an ongoing issue where the `Gres` parameter on the GPU nodes is not being set properly across all the nodes, and seems to change value over time, currently with an unknown root cause. While the right number of GPUs are still assigned when this happens, the same is not true for the number of CPUs. This leads to users having access to the wrong number of CPU cores based on this resource request and the subsequent account billing to also be incorrect. The single test parameter - the number of GPUs per node - can be modified as desired.

### `gpu_affinity_check`

A test for correct binding of processes to CPUs and associated GPUs. The test checks that processes are bound to separate GPUs and that the CPUs they are bound to are valid CPUs for the assigned GPU. The job options can be modified as desired. The test parameter sets the type of job access, exclusive or shared.

### `gpu_affinity_array_check`

This test is mostly the same as `gpu_affinity_check` except that the job is not a standard exclusive or shared job with a single `srun` statement. The job in this test uses a job array. The test parameter `job_config` sets the number of nodes, number of array tasks, GPUs per node, and tasks per node. A variety of default values are provided, but these can be modified as desired. The job options can also be modified as needed.

### `gpu_affinity_job_packing_check`

Analogous to `gpu_affinity_array_check`, except now the job utilises job packing via job steps. All job options can be modified, as can the test parameter `job_config`, which sets the number of nodes, the number of GPUs per node for each of the two job steps, and the number of GPUs per task.

### `gpu_accounting_check`

This test checks that SLURM accounting is being calculated corectly on GPU nodes. It runs a simple job which simply sleeps for a certain amount of time and then check that the accounting from `sacct` matches what it should given the job's resource allocation request.