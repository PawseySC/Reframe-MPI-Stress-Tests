# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

import sys
import os.path

# Add root directory of repo to path
curr_dir = os.path.dirname(__file__).replace('\\','/')
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)
# Import functions to set env vars, modules, commands
from common.scripts.parse_yaml import *
config_path = curr_dir + '/gpu_compilation_config.yaml'


class gpu_compile_base_check(rfm.CompileOnlyRegressionTest, pin_prefix = True):
    def __init__(self, name, **kwargs):

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Metadata
        self.descr = 'Base class for compiling warmup and multi-GPU tests from performance-modelling-tools repo'
        self.maintainers = ['Craig Meyer']

        # Build from makefile, using options specified in github repo
        self.build_system = 'Make'
        self.build_system.flags_from_environ = False


@rfm.simple_test
class warmup_compile_check(gpu_compile_base_check):
    def __init__(self, **kwargs):
        super().__init__('warmup_compile_check', **kwargs)

        # Metadata
        self.descr = 'Test class for compiling warmup test from performance-modelling-tools repo'
        self.maintainers = ['Craig Meyer']

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        self.prebuild_cmds = [f'make -f Makefile_warmup clean',]
        self.build_system.options = ['-f Makefile_warmup', f'buildtype={self.build_type}']
        

    # Test parameters
    params = get_test_params(config_path, 'warmup_compile_check')
    build_type = parameter(params['build-type'])

    # Test passes if compiled executable exists
    @sanity_function
    def assert_compiled(self):
        return sn.assert_true(os.path.exists(f'./warm_up_test.{self.build_type}.exe'))


@rfm.simple_test
class multigpu_compile_check(gpu_compile_base_check):
    def __init__(self, **kwargs):
        super().__init__('multigpu_compile_check', **kwargs)

        # Metadata
        self.descr = 'Test class for compiling multigpu test from performance-modelling-tools repo'
        self.maintainer = ['Craig Meyer']

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        self.prebuild_cmds = [f'make -f Makefile_multigpu clean']
        self.build_system.options = ['-f Makefile_multigpu', f'buildtype={self.build_type}']

    # Test parameter
    params = get_test_params(config_path, 'multigpu_compile_check')
    build_type = parameter(params['build-type'])

    # Test passes if compiled executable exists
    @sanity_function
    def assert_compiled(self):
        return sn.assert_true(os.path.exists(f'./multigpu.{self.build_type}.exe'))


# Probably can form a base class, and separate tests for warmup, performance, and offloading tests
@rfm.simple_test
class warmup_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node'] # Maximum number of GPUs available per node
        self.cpus_per_gpu = sys_info['cpus-per-gpu'] # Number of CPUs associated with each GPU
        job_info = get_job_options(config_path)
        self.acct_str = job_info['account']

        # Metadata
        self.descr = 'Test GPU warmup times using test from performance-modelling-tools repo'
        self.maintainers = ['Craig Meyer']

        # Job options (these will automatically be set in the job script by ReFrame)
        self.num_tasks = self.ngpus_per_node
        self.ncpus_per_task = self.cpus_per_gpu
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True

        test_config = configure_test(config_path, 'warmup_check')

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        self.variables['OMP_NUM_THREADS'] = str(self.ncpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        
        self.keep_files = ['logs/*']
        
        # Performance reference values to check against
        self.scaling_factor = self.ngpus_per_node
        if self.build_type == 'hip':
            ref_dict = test_config['performance']['hip']['reference-value']
            self.reference = {
                '*': {key: (val * self.scaling_factor**2, None, 0.2) for key, val in ref_dict.items()}
            }
        elif 'omp' in self.build_type:
            ref_dict = test_config['performance']['omp']['reference-value']
            self.reference = {
                '*': {key: (val * self.scaling_factor**2, None, 0.2) for key, val in ref_dict.items()}
            }

        self.tags = {'gpu'}


    # Test parameters
    params = get_test_params(config_path, 'warmup_check')
    ngpus_per_node = parameter(params['ngpus-per-node'])
    build_type = parameter(params['build-type'])

    # Test dependency - this test depends on the compilation test and only runs if that test passes,
    # so failure can immediately be diagnosed as compilation or performance/execution
    @run_after('init')
    def inject_dependencies(self):
        testdep_name = f'warmup_compile_check_{self.build_type}'
        self.depends_on(testdep_name, udeps.by_env)
    @run_after('setup')
    def set_executable(self):
        td = self.getdep(f'warmup_compile_check_{self.build_type}')
        self.sourcedir = td.stagedir
        self.executable = f'warm_up_test.{self.build_type}.exe'
        self.prerun_cmds += [
            f'cd {self.sourcedir}',
            'export LD_LIBRARY_PATH=$(pwd)/common/performance-modelling-tools/tests/profile_util/lib/:$LD_LIBRARY_PATH',
        ]
    # Performance variables
    @run_before('performance')
    def set_perf_dict(self):
        self.perf_variables = {
            'run_kernel': self.extract_timing(),
            'alloc': self.extract_timing('alloc'),
            'free': self.extract_timing('free'),
            'kernel': self.extract_timing('kernel'),
            'd2h': self.extract_timing('d2h'),
            'h2d': self.extract_timing('h2d'),
        }
    # Performance function - extract time measurements from code logging
    @performance_function('us')
    def extract_timing(self, kind = 'run_kernel'):
        
        times = sn.evaluate(sn.extractall(r'Times \(ave,stddev\) \[us\] = \(([0-9]+.[0-9]+),.*\)', self.stdout, 1, float))
        ntimes = len(times)
        
        if kind == 'run_kernel':
            return sum(times[0:ntimes:6]) / self.ngpus_per_node
        elif kind == 'alloc':
            return sum(times[1:ntimes:6]) / self.ngpus_per_node
        elif kind == 'free':
            return sum(times[2:ntimes:6]) / self.ngpus_per_node
        elif kind == 'kernel':
            return sum(times[3:ntimes:6]) / self.ngpus_per_node
        elif kind == 'd2h':
            return sum(times[4:ntimes:6]) / self.ngpus_per_node
        elif kind == 'h2d':
            return sum(times[5:ntimes:6]) / self.ngpus_per_node

    # Job options
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--account={self.acct_str}',
            f'--gres=gpu:{self.ngpus_per_node}'
        ]

    # Sanity function
    @sanity_function
    def assert_warmup(self):
        return sn.assert_found('Reporting times for', self.stdout)


# Regression test for multigpu test in performance-modelling-tools repo
@rfm.simple_test
class multigpu_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test performance with multiple GPUs using test from performance-modelling-tools repo'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node'] # Maximum number of GPUs available per node
        self.cpus_per_gpu = sys_info['cpus-per-gpu'] # Number of CPUs associated with each GPU
        job_info = get_job_options(config_path)
        self.acct_str = job_info['account']

        test_config = configure_test(config_path, 'multigpu_check')

        # Job options (these will automatically be set in the job script by ReFrame)
        self.num_tasks = self.ngpus_per_node
        self.ncpus_per_task = self.cpus_per_gpu
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        self.variables['OMP_NUM_THREADS'] = str(self.ncpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        
        self.keep_files = ['logs/*']

        # Performence reference values to check against
        self.scaling_factor = self.ngpus_per_node
        if self.build_type == 'hip':
            ref_dict = test_config['performance']['hip']['reference-value']
            self.reference = {
                '*': {key: (val * self.scaling_factor**2, None, 0.2) for key, val in ref_dict.items()}
            }

        self.tags = {'gpu'}

    # Test parameters
    params = get_test_params(config_path, 'multigpu_check')
    ngpus_per_node = parameter(params['ngpus-per-node'])
    build_type = parameter(params['build-type'])

    # Test dependency - this test depends on the compilation test and only runs if that test passes,
    # so failure can immediately be diagnosed as compilation or performance/execution
    @run_after('init')
    def inject_dependencies(self):
        testdep_name = f'multigpu_compile_check_{self.build_type}'
        self.depends_on(testdep_name, udeps.by_env)
    @run_after('setup')
    def set_executable(self):
        td = self.getdep(f'multigpu_compile_check_{self.build_type}')
        self.sourcedir = td.stagedir
        self.executable = f'multigpu.{self.build_type}.exe'
        self.prerun_cmds = [
            f'cd {self.sourcedir}',
            'export LD_LIBRARY_PATH=$(pwd)/common/performance-modelling-tools/tests/profile_util/lib/:$LD_LIBRARY_PATH',
        ]
    # Performance variables
    @run_before('performance')
    def set_perf_dict(self):
        self.perf_variables = {
            'run_kernel': self.extract_timing(),
            'kernel': self.extract_timing('kernel'),
            'd2h': self.extract_timing('d2h'),
            'h2d': self.extract_timing('h2d'),
        }
    # Performance function - extract time measurements from code logging
    @performance_function('us')
    def extract_timing(self, kind = 'run_kernel'):
        
        times = sn.evaluate(sn.extractall(r'Times \(ave,stddev\) \[us\] = \(([0-9]+.[0-9]+),.*\)', self.stdout, 1, float))
        ntimes = len(times)
        
        if kind == 'run_kernel':
            return sum(times[0:ntimes:4]) / self.ngpus_per_node
        elif kind == 'kernel':
            return sum(times[1:ntimes:4]) / self.ngpus_per_node
        elif kind == 'd2h':
            return sum(times[2:ntimes:4]) / self.ngpus_per_node
        elif kind == 'h2d':
            return sum(times[3:ntimes:4]) / self.ngpus_per_node

    # Job options
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [f'--account={self.acct_str}', f'--gres=gpu:{self.ngpus_per_node}']

    # Sanity function
    @sanity_function
    def assert_multigpu(self):
        return sn.assert_found('Reporting times for', self.stdout)
