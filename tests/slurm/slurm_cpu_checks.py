# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

from math import ceil

import sys
import os.path

# Add root directory of repo to path
curr_dir = os.path.dirname(__file__).replace('\\','/')
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)
# Import functions to set env vars, modules, commands
from common.scripts.parse_yaml import *
config_path = curr_dir + '/slurm_cpu_config.yaml'

@rfm.simple_test
class MemoryCompileTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Compile code used for memory tests'
        self.maintaines = ['Craig Meyer']

        sys_info = set_system(config_path, 'MemoryCompile')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        # Account to charge job to
        job_info = get_job_options(config_path, 'MemoryCompile')
        self.acct_str = job_info['account']

        # Compilation and execution
        self.build_system = 'SingleSource'
        # Build profile_util library needed for cdode
        self.prebuild_cmds = [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util', 
            './build_cpu.sh', 
            'PROFILE_UTIL_DIR=$(pwd)', 
            'cd ${MAIN_SRC_DIR}',
        ]
        self.build_system.cppflags = [
            '-fopenmp', '-O3', '-D_MPI',
            '-L${PROFILE_UTIL_DIR}/lib',
            '-I${PROFILE_UTIL_DIR}/include',
            '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/',
            '-lprofile_util_mpi_omp',
        ]
        self.sourcepath = './mem_report.cpp'
        self.executable = 'mem_report.out'

        self.tags = {'slurm'}
        
    # Test passes if compiled executable exists
    @sanity_function
    def assert_compiled(self):
        return sn.assert_true(os.path.exists('mem_report.out'))

@rfm.simple_test
class slurm_node_mem_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check memory per node allocated in SLURM matches resource request'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path, 'slurm_node_mem_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.max_mem_per_cpu = sys_info['max-mem-per-cpu']

        # Job options
        job_info = get_job_options(config_path, 'slurm_node_mem_check')
        self.acct_str = job_info['account']
        self.num_nodes = job_info['num-nodes']
        self.num_tasks_per_node = job_info['num-tasks-per-node']
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_cpus_per_task = job_info['num-cpus-per-task']
        self.num_tasks_per_core = job_info['num-tasks-per-core']
        if self.multithreading:
            self.nthreads_per_core = 2
            self.mem_per_cpu = self.max_mem_per_cpu // self.nthreads_per_core
        else:
            self.nthreads_per_core = 1
            self.mem_per_cpu = self.max_mem_per_cpu
        self.mem_per_node = self.num_tasks_per_node * self.num_cpus_per_task * self.mem_per_cpu * self.nthreads_per_core // self.num_tasks_per_core
        
        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Check value of SLURM environment variable
        self.postrun_cmds = ['echo Each node has been allocated $SLURM_MEM_PER_NODE MB of memory']

        self.tags = {'slurm'}

    # Test parameter(s)
    params = get_test_params(config_path, 'slurm_node_mem_check')
    multithreading = parameter(params['multithreading'])

    # This test depends on `MemoryCompileTest`
    @run_after('init')
    def inject_dependencies(self):
        self.depends_on('MemoryCompileTest', udeps.by_env)
    @require_deps
    def set_executable(self, MemoryCompileTest):
        self.sourcedir = MemoryCompileTest().stagedir
        self.executable = os.path.join(self.sourcedir, 'mem_report.out')
        # Call program with 95% of --mem-per-node to allow for a margin of error
        self.executable_opts = ['%0.f' % (self.mem_per_node * 0.95)]
    
    # Set job options
    @run_before('run')
    def set_job_options(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--threads-per-core={self.nthreads_per_core}',
            f'--mem={self.mem_per_node}',
            f'--account={self.acct_str}',
        ]
    # Run memory reporting with 1 task per node before main `srun` 
    # call so that all requested memory per node is available to the task
    @run_before('run')
    def run_one_task(self):
        cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [
            f'{cmd} -N {self.num_nodes} -n {self.num_nodes} -c {self.num_cpus_per_task} {self.executable} {self.executable_opts[0]}',
        ]
    # Explicitly set -c in srun statements (needed for SLURM > 21.08)
    @run_before('run')
    def set_cpus_per_task(self):
        if self.job.scheduler.registered_name in ['slurm', 'squeue']:
            self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
    
    # Test passes if there is one success and one failure and if the SLURM variable is correct
    # One success should be from `--ntasks=--nodes` run
    # One failure should be from `--ntasks=--ntasks-per-node` run
    @sanity_function
    def assert_valid_mem(self):
        return sn.all([
            sn.assert_found(f'Each node has been allocated {self.mem_per_node} MB of memory', self.stdout),
            sn.assert_found('Memory report @', self.stdout),
            sn.assert_found('Out Of Memory', self.stderr),
        ])

@rfm.simple_test
class slurm_cpu_mem_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check that memory per CPU allocated in SLURM matches resource request'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path, 'slurm_cpu_mem_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.max_mem_per_cpu = sys_info['max-mem-per-cpu']

        # Job options
        job_info = get_job_options(config_path, 'slurm_cpu_mem_check')
        self.acct_str = job_info['account']
        self.num_nodes = job_info['num-nodes']
        self.num_tasks = job_info['num-tasks']
        self.num_tasks_per_core = job_info['num-tasks-per-core']
        self.num_cpus_per_task = job_info['num-cpus-per-task']
        if self.multithreading:
            self.nthreads_per_core = 2
            self.mem_per_cpu = self.max_mem_per_cpu // self.nthreads_per_core
        else:
            self.nthreads_per_core = 1
            self.mem_per_cpu = self.max_mem_per_cpu
        self.mem_per_node = self.num_tasks * self.num_cpus_per_task * self.mem_per_cpu * self.nthreads_per_core // self.num_tasks_per_core

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Output job resource allocation info via `scontrol` and memory reporting via `seff`
        self.postrun_cmds = ['echo Each CPU has been allocated $SLURM_MEM_PER_CPU MB of memory']

        self.tags = {'slurm'}

    # Test parameter(s)
    params = get_test_params(config_path, 'slurm_cpu_mem_check')
    multithreading = parameter(params['multithreading'])

    # This test depends on `MemoryCompileTest`
    @run_after('init')
    def inject_dependencies(self):
        self.depends_on('MemoryCompileTest', udeps.by_env)
    @require_deps
    def set_executable(self, MemoryCompileTest):
        self.sourcedir = MemoryCompileTest().stagedir
        self.executable = os.path.join(self.sourcedir, 'mem_report.out')
        # Call program with 95% of --mem-per-node to allow for a margin of error
        self.executable_opts = ['%0.f' % (self.mem_per_node * 0.95)]

    # Job options
    @run_before('run')
    def set_job_options(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--threads-per-core={self.nthreads_per_core}',
            f'--mem-per-cpu={self.mem_per_cpu}',
            f'--account={self.acct_str}',
        ]
    # Run memory reporting with 1 task before main `srun` 
    # call so that all requested memory per node is available
    @run_before('run')
    def run_one_task(self):
        cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds += [f'{cmd} -n 1 {self.executable} {self.executable_opts[0]}']
    # Explicitly set -c in srun statements (needed for SLURM > 21.08)
    @run_before('run')
    def set_cpus_per_task(self):
        if self.job.scheduler.registered_name in ['slurm', 'squeue']:
            self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
    
    # Test passes if there is one success and one failure, and if the slurm variable is correct
    # One success should be from --ntasks=1 run
    # One failure should be from `--ntasks=--ntasks` run
    @sanity_function
    def assert_valid_mem(self):
        return sn.all([
            sn.assert_found(f'Each CPU has been allocated {self.mem_per_cpu} MB of memory', self.stdout),
            sn.assert_found('Memory report @', self.stdout),
            sn.assert_found('Out Of Memory', self.stderr)
        ])




@rfm.simple_test
class slurm_billing_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check that the billed CPUs are correct given the resource request'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        sys_info = set_system(config_path, 'slurm_billing_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.max_mem_per_cpu = sys_info['max-mem-per-cpu']

        # Executable here is meaningless, it just allows the test to run
        self.executable = 'echo'
        self.executable_opts = ['hello world']

        # Job options
        job_info = get_job_options(config_path, 'slurm_billing_check')
        self.acct_str = job_info['account']
        self.num_nodes = job_info['num-nodes']
        self.num_tasks_per_node = job_info['num-tasks-per-node']
        self.num_tasks = self.num_tasks_per_node * self.num_nodes
        self.num_cpus_per_task = self.ncpus_per_task
        self.mem_per_node = 8 * self.max_mem_per_cpu
        # Set the memory per cpu to `max_mem_per_cpu` to get the number of CPUs needed for a given memory
        self.mem_per_cpu = self.max_mem_per_cpu

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        self.tags = {'slurm'}

    # Test parameter(s)
    params = get_test_params(config_path, 'slurm_billing_check')
    multithreading = parameter(params['multithreading'])
    ncpus_per_task = parameter(params['num-cpus-per-task'])

    # Job options
    @run_before('run')
    def set_job_opts(self):
        if self.multithreading:
            self.job.options = ['--threads-per-core=2']
        else:
            self.job.options = ['--threads-per-core=1']
        self.job.options += [
            f'--mem={self.mem_per_node}',
            f'--account={self.acct_str}',
        ]

    # Calculate resources that should be charged for this job given resource request and system setup
    @run_before('run')
    def calc_resources(self):
        self.proc = self.current_partition.processor
        self.num_cpus_per_node = self.proc.num_cpus
        self.num_cpus_per_core = self.proc.num_cpus_per_core
        if not self.multithreading:
            self.num_cpus_per_node //= self.num_cpus_per_core
            self.num_cpus_per_core = 1
            # Factor to convert `--cpus-per-task` and `--threads-per-core` to actual no. of cores available
            self.cores_to_cpus = self.num_cpus_per_task
        # If hyperthreading and `--cpus-per-task` is even, no factor needed
        # Otherwise, the last CPU of each task is placed on one core of a CPU,
        # but the other core is not utilised in the next task. This leads to "hidden"
        # extra cores available to use
        else:
            if self.num_cpus_per_task % self.num_cpus_per_core == 0:
                self.cores_to_cpus = self.num_cpus_per_task
            else:
                self.cores_to_cpus = (self.num_cpus_per_task // self.num_cpus_per_core + 1) * self.num_cpus_per_core
        self.max_cores = self.num_nodes * self.num_tasks_per_node * self.cores_to_cpus
    
    # Test if the billing values seen in `scontrol` output
    # match what they should be given the resource allocation request.
    @sanity_function
    def assert_billing(self):
        req_cpus = self.cores_to_cpus * self.num_tasks_per_node * self.num_nodes
        req_mem = self.mem_per_node * self.num_nodes
        mem_cpus = ceil(req_mem / self.mem_per_cpu) * 2
        # If `--threads-per-core=1`, we are actually charged twice the no. of CPUs
        # then we asked for, since we don't have access to the SMT on each CPU
        if not self.multithreading:
            req_cpus *= 2
        max_billing = max([req_cpus, mem_cpus])

        billing = sn.extractsingle(r'billing=(?P<billing>\S+)', self.stdout, 'billing', int)
        num_cpus = sn.extractsingle(r'NumCPUs=(?P<num_cpus>\S+)', self.stdout, 'num_cpus', int)
        memory = sn.extractsingle(r'mem=(?P<memory>\S+)M.+', self.stdout, 'memory', int)
        tres_cpu = sn.extractsingle(r'TRES=cpu=(?P<tres_cpu>\S+),mem.+', self.stdout, 'tres_cpu', int)

        return sn.all([
            sn.assert_eq(max_billing, billing),
            sn.assert_eq(billing, num_cpus),
            sn.assert_eq(tres_cpu, billing),
            sn.assert_eq(req_mem, memory)
        ])



@rfm.simple_test
class slurm_cpu_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check the number of cores requested in a job matches what is allocated'
        self.maintainers = ['Craig', 'Pascal Jahan Elahi']

        sys_info = set_system(config_path, 'slurm_cpu_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        
        # Executable - meaningless, needed for test to run
        self.executable = 'echo'
        self.executable_opts = ['hello world']

        # Job options
        job_info = get_job_options(config_path, 'slurm_cpu_check')
        self.acct_str = job_info['account']
        self.num_nodes = job_info['num-nodes']
        self.num_cpus_per_task = job_info['num-cpus-per-task']
        self.num_tasks_per_node = job_info['num-tasks-per-node']
        self.num_tasks = self.num_tasks_per_node * self.num_nodes

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Check value of SLURM environment variable
        self.postrun_cmds = ['echo There are $SLURM_JOB_CPUS_PER_NODE CPUs allocated per node']

        self.tags = {'slurm'}

    # Test parameter(s)
    params = get_test_params(config_path, 'slurm_cpu_check')
    multithreading = parameter(params['multithreading'])
    
    # Job options
    @run_before('run')
    def set_job_options(self):
        if self.multithreading:
            self.job.options = ['--threads-per-core=2']
        else:
            self.job.options = ['--threads-per-core=1']
        self.job.options += [
            f'--nodes={self.num_nodes}',
            f'--account={self.acct_str}',
        ]

        
    # Calculate resources that should be charged for this job given resource request and system setup
    @run_before('run')
    def calc_resources(self):
        self.proc = self.current_partition.processor
        self.num_cpus_per_node = self.proc.num_cpus
        self.num_cpus_per_core = self.proc.num_cpus_per_core
        # Factor to convert `--cpus-per-task` and `--threads-per-core` to #actual number of 
        # cores that are available. Will change if in certain hyperthreading setups
        self.cores_to_cpus = self.num_cpus_per_task
        if not self.multithreading:
            self.num_cpus_per_node //= self.num_cpus_per_core
            self.num_cpus_per_core = 1
        # If hyperthreading and `--cpus-per-task` is odd, the last CPU of each task
        # is placed on one core of a CPU, but the other core on that CPU is not utilised
        # by the next task. This leads to "hidden" extra cores available to use
        else:
            if self.num_cpus_per_task % self.num_cpus_per_core != 0:
                self.cores_to_cpus = (self.num_cpus_per_task // self.num_cpus_per_core + 1) * self.num_cpus_per_core
        self.max_cores = self.num_nodes * self.num_tasks_per_node * self.cores_to_cpus

    # Call `srun` at maximum core limit and one core higher
    # This should result in one success and one failure
    @run_before('run')
    def pre_launch(self):
        cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds += [
            f'{cmd} -n 1 -c {c} {self.executable} {self.executable_opts[0]}'
            for c in range(self.max_cores, self.max_cores + 2)
        ]

    # Test passes if the --cpus-per-task=max_cores run succeeds
    # (outputting the value 1), and the --cpus-per-task=max_cores+1
    # run fails, and that the number of CPUs per node env var is correct
    @sanity_function
    def assert_no_overload(self):
        return sn.all([
            sn.assert_found(r'.{0}1.{0}', self.stdout),
            sn.assert_eq(sn.extractsingle(r'There are\s+(?P<ncpus>\(*x*[0-9]*\)*)\(*x*[0-9]*\)* CPUs allocated per node',
                                          self.stdout, 'ncpus', int), self.max_cores // self.num_nodes),
            sn.assert_found(r'More processors requested than permitted', self.stderr)
        ])

@rfm.simple_test
class omp_thread_check(rfm.RegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test that number of OMP threads available equals no. of CPUs requested'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path, 'omp_thread_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Compilation
        self.build_system = 'SingleSource'
        self.sourcepath = './affinity_report.cpp'
        # Build proflie_util library needed by code
        self.prebuild_cmds += [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util', 
            './build_cpu.sh', 
            'PROFILE_UTIL_DIR=$(pwd)', 
            'cd ${MAIN_SRC_DIR}',
        ]
        self.build_system.cppflags = [
            '-fopenmp', '-O3', '-D_MPI', 
            '-L${PROFILE_UTIL_DIR}/lib', 
            '-I${PROFILE_UTIL_DIR}/include', 
            '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/', 
            '-lprofile_util_mpi_omp',
            ]
        # Executable is thread affinity reporting program
        self.executable = 'affinity_report.out'
        
        # Job options
        job_info = get_job_options(config_path, 'omp_thread_check')
        self.acct_str = job_info['account']
        self.num_cpus_per_task = job_info['num-cpus-per-task']

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        self.variables['OMP_DISPLAY_ENV'] = 'VERBOSE'
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        
        self.tags = {'slurm'}

    # Test parameter(s)
    params = get_test_params(config_path, 'omp_thread_check')
    multithreading = parameter(params['multithreading'])
    cpu_binding = parameter(params['cpu-binding'])

        
    # Explicitly set -c in srun statements (needed for SLURM > 21.08)
    # Also set `--cpu-bind` option since it can affect the no. of threads
    @run_before('run')
    def modify_launcher(self):
        self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
        if self.cpu_binding != 'auto':
            self.job.launcher.options += [f'--cpu-bind={self.cpu_binding}']
    # Job options
    @run_before('run')
    def set_job_options(self):
        if self.multithreading:
            self.job.options = ['--threads-per-core=2']
        else:
            self.job.options = ['--threads-per-core=1']
        self.job.options += [f'--account={self.acct_str}']

    # Test passes if the no. of threads reported from `aff_report.out` 
    # program matches the value of `--cpus-per-task`
    @sanity_function
    def assert_thread_count(self):
        self.threads_per_core = 2 if self.multithreading else 1

        nthreads = sn.extractsingle(r'OpenMP version.+total number of threads\s+=\s+(?P<nthreads>\S+).+',
                                    self.stdout, 'nthreads', int)
        return sn.assert_eq(nthreads, self.num_cpus_per_task * self.threads_per_core)


# Test to compile a C++ program that records the OMP thread affinity
# (and MPI rank affinity if there is more than one task)
@rfm.simple_test
class AffinityCompileTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test class for compiling an OMP thread/MPI rank affinity reporting program'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path, 'AffinityCompile')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        job_info = get_job_options(config_path, 'AffinityCompile')
        self.acct_str = job_info['account']

        # Compilation
        self.build_system = 'SingleSource'
        self.sourcepath = './affinity_report.cpp'
        # Build profile util library
        self.prebuild_cmds += [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util', 
            './build_cpu.sh', 
            'PROFILE_UTIL_DIR=$(pwd)', 
            'cd ${MAIN_SRC_DIR}',
        ]
        self.build_system.cppflags = [
            '-fopenmp', '-D_MPI', '-O3',
            '-L${PROFILE_UTIL_DIR}/lib',
            '-I${PROFILE_UTIL_DIR}/include',
            '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/',
            '-lprofile_util_mpi_omp',
        ]
        self.executable = 'affinity_report.out'

        self.tags = {'slurm'}
        
    # Test passes if compiled executable exists
    @sanity_function
    def assert_compiled(self):
        return sn.assert_true(os.path.exists('affinity_report.out'))
    

# Test to run a compiled C++ executable that records the OMP thread affinity
# (and MPI rank affinity if there is more than one task)
@rfm.simple_test
class affinity_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test class for running and analysing an OMP thread/MPI rank affinity reporting program'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path, 'affinity_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.max_mem_per_cpu = sys_info['max-mem-per-cpu']

        # Execution
        self.executable_opts = ['&> affinity.txt']
        # Keep txt files holding affinity information
        self.keep_files = ['affinity.txt']

        # Job options
        job_info = get_job_options(config_path, 'affinity_check')
        self.acct_str = job_info['account']
        self.num_cpus_per_task = job_info['num-cpus-per-task']
        self.num_nodes = job_info['num-nodes']
        if self.access == 'exclusive':
            self.num_tasks_per_node = 16
            self.exclusive_access = True
        elif self.access == 'shared':
            self.num_tasks_per_node = 2
        if self.multithreading:
            self.nthreads_per_core = 2
            self.mem_per_cpu = self.max_mem_per_cpu // self.nthreads_per_core
        else:
            self.nthreads_per_core = 1
            self.mem_per_cpu = self.max_mem_per_cpu
        self.num_tasks = self.num_tasks_per_node * self.num_nodes

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        self.omp_num_threads = self.num_cpus_per_task
        self.variables['OMP_NUM_THREADS'] = str(self.omp_num_threads)
        self.variables['OMP_DISPLAY_AFFINITY'] = 'TRUE'
        self.variables['OMP_DISPLAY_ENV'] = 'VERBOSE'
        # Specify distribution of OMP threads
        self.variables['OMP_PROC_BIND'] = self.omp_proc_bind
        self.variables['OMP_PLACES'] = self.omp_places
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        self.tags = {'slurm'}
    
    # Test parameters
    params = get_test_params(config_path, 'affinity_check')
    access = parameter(params['access'])
    omp_proc_bind = parameter(params['omp-proc-bind'])
    omp_places = parameter(params['omp-places'])
    multithreading = parameter(params['multithreading'])

    # This test depends on `AffinityCompileTest`
    @run_after('init')
    def inject_dependencies(self):
        self.depends_on('AffinityCompileTest', udeps.by_env)
    @require_deps
    def set_executable(self, AffinityCompileTest):
        self.sourcedir = AffinityCompileTest().stagedir
        self.executable = os.path.join(self.sourcedir, 'affinity_report.out')

    # Job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--account={self.acct_str}',
            f'--threads-per-core={self.nthreads_per_core}',
        ]
    # Explicitly set -c in srun statements (needed for SLURM > 21.08)
    @run_before('run')
    def srun_cpus_per_task(self):
        if self.job.scheduler.registered_name in ['slurm', 'squeue']:
            self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
    # Get processor specs - needed to correctly interpret affinity output
    # The specs need to be added to `settings.py` for this to work
    # NOTE: All values are set assuming hyperthreading (--threads-per-core=2)
    @run_before('run')
    def get_proc_info(self):
        self.proc = self.current_partition.processor
        self.num_cpus = self.proc.num_cpus
        self.num_cpus_per_core = self.proc.num_cpus_per_core
        self.num_cpus_per_socket = self.proc.num_cpus_per_socket
        self.num_sockets = self.proc.num_sockets
    # Define postrun commands to call python affinity analysis script
    @run_before('run')
    def analyse_affinity(self):
        self.postrun_cmds = [
            'python3 parse_affinity.py ' + 
            f'-N 1 ' +
            f'-p {self.num_tasks},{self.multithreading},{self.omp_num_threads},{self.omp_proc_bind},{self.omp_places},{self.num_cpus_per_task} ' +
            f'-s {self.num_sockets},{self.num_cpus},{self.num_cpus_per_core},{self.num_cpus_per_socket} ' +
            '-f affinity.txt -m OMP'
        ]
        self.postrun_cmds += [
            'python3 parse_affinity.py ' + 
            f'-N {self.num_nodes} ' +
            f'-p {self.num_tasks},{self.multithreading},{self.omp_num_threads},{self.omp_proc_bind},{self.omp_places},{self.num_cpus_per_task} ' +
            f'-s {self.num_sockets},{self.num_cpus},{self.num_cpus_per_core},{self.num_cpus_per_socket} ' +
            '-f affinity.txt -m MPI'
        ]

    # Test passes if compiled program runs to completion and
    # OMP affinity and MPI affinity are both good
    @sanity_function
    def check_affinity(self):

        return sn.all([
            sn.assert_found(r'Affinity report finished!', 'affinity.txt'),
            # OMP affinity conditions
            sn.assert_found(r'OMP THREAD AFFINITY.+SENSIBLE!', self.stdout),
            sn.assert_not_found(r'OMP THREAD AFFINITY.+UNEXPECTED!', self.stdout),
            # MPI affinity conditions
            sn.assert_found(r'MPI RANK AFFINITY.+SENSIBLE!', self.stdout),
            sn.assert_not_found(r'MPI RANK AFFINITY.+UNEXPECTED!', self.stdout),
        ])

# heterogeneous job test
@rfm.simple_test
class het_job_test(rfm.RegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Testing heterogeneous MPI jobs in SLURM'
        self.maintainers = ['Craig Meyer', 'pascal.elahi@pawsey.org.au']
        
        sys_info = set_system(config_path, 'het_job_test')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.max_mem_per_cpu = sys_info['max-mem-per-cpu']

        # Compilation - build from makefile
        self.build_system = 'Make'
        self.build_system.flags_from_environ = False
        # Build profile_util library
        self.prebuild_cmds += [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util', './build_cpu.sh', 'PROFILE_UTIL_DIR=$(pwd)', 
            'cd ${MAIN_SRC_DIR}',
            'make'
        ]
        self.executable = 'mpicomm'
        self.executable_opts = ['1> hetjob.log 2> hetjob.err']

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        job_info = get_job_options(config_path, 'het_job_test')
        self.acct_str = job_info['account']
        self.time_limit = job_info['time-limit']

        # Modify srun options depending on mode of heterogeneous job submission
        if self.mode == 'srun':
            self.prerun_cmds += [
                f'srun --ntasks=4 --ntasks-per-node=2 {self.executable} 1> nonhetjob.log 2> nonhetjob.err'
        ]
        elif self.mode == 'sbatch':
            self.prerun_cmds += [
                f'srun --ntasks-per-node=2 {self.executable} 1> nonhetjob.log 2> nonhetjob.err',
                f'srun --het-group=0 {self.executable} 1> hetgroup0.log 2> hetgroup0.err',
                f'srun --het-group=1 {self.executable} 1> hetgroup1.log 2> hetgroup1.err',
            ]
        elif self.mode =='multiprog':
            self.prerun_cmds += [
                f'srun --multi-prog multiprog.conf 1> nompi-multiprog.log 2> nompi-multiprog.err',
            ]

        self.tags = {'slurm'}

    # Test parameters
    params = get_test_params(config_path, 'het_job_test')
    mode = parameter(params['mode'])


    # Set srun options depending on mode of heterogeneous job submission
    @run_before('run')
    def run_het_jobs(self):
        if self.mode == 'srun':
            self.job.launcher.options = [
                '--export=ALL -c 1 -n 6 --ntasks-per-node=6 --mem=20GB : -c 16 -n 1 --ntasks-per-node=1 --mem=10GB',
            ]
        elif self.mode == 'sbatch':
            self.job.launcher.options = ['--het-group=0,1']
        elif self.mode == 'multiprog':
            self.job.launcher.options = ['--multi-prog']
            self.executable = 'multiprog.mpi.conf'
            self.executable_opts= ['1> mpi-multiprog.log 2> mpi-multiprog.err']


    # Job options depend on heterogeneous job mode
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [f'--account={self.acct_str}']
        if self.mode == 'srun':
            self.job.options += [
                '--nodes=2',
                '--mem=0',
                '--ntasks=256',
            ]
        elif self.mode == 'sbatch':
            self.job.options += [
                f'-c 1 -p {self.current_partition.name} -n 6 --ntasks-per-node=6 --mem=20GB',
                'hetjob',
                f'-c 16 -p {self.current_partition.name} -n 1 --ntasks-per-node=1 --mem=10GB',
            ]
        elif self.mode == 'multiprog':
            self.job.options += [
                '--nodes=2',
                '--ntasks=8',
                '--ntasks-per-node=4',
                '--mem=20GB',
            ]

    # Test passes if job runs to completion. The form completion takes depends on mode
    @sanity_function
    def assert_het_job(self):
        # Test will pass if code runs to completion
        if self.mode == 'sbatch':
            return sn.all([
                sn.assert_found('Ending job', 'hetjob.log'),
                sn.assert_found('Ending job', 'hetgroup0.log'),
                sn.assert_found('Ending job', 'hetgroup1.log'),
            ])
        elif self.mode == 'srun':
            return sn.assert_found('Ending job', 'hetjob.log')
        elif self.mode == 'multiprog':
            return sn.assert_found('Ending job', 'mpi-multiprog.log')


@rfm.simple_test
class accounting_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        sys_info = set_system(config_path, 'accounting_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Execution - sleep for 2 minutes
        self.executable = 'sleep'
        self.executable_opts = ['120s']

        # Job options
        job_info = get_job_options(config_path, 'accounting_check')
        self.acct_str = job_info['account']
        self.num_nodes = job_info['num-nodes']
        self.num_tasks = job_info['num-tasks']
        self.num_cpus_per_task = job_info['num-cpus-per-task']
        self.time_limit = job_info['time-limit']

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Extract job accounting information with sacct
        self.postrun_cmds = ['sacct -X --name rfm_accounting_check_job --format=CPUTimeRaw | grep -v batch | awk \'{sum=$1}END{print sum/3600/2}\'']


    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--account={self.acct_str}'
        ]

    @sanity_function
    def assert_correct_accounting(self):
        return sn.assert_found(r'0.4', self.stdout)