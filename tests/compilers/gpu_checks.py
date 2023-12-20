# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

# Import functions to set environment variables, etc.
import sys
import os
sys.path.append(os.getcwd() + '/common/scripts')
from set_test_env import *


class gpu_compile_base_check(rfm.CompileOnlyRegressionTest, pin_prefix = True):
    def __init__(self, name, **kwargs):

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['*']

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
        _, modules, _ = set_env(mpi = False, gpu = True)
        if modules != []:
            self.modules = modules
        self.prebuild_cmds = [f'make -f Makefile_warmup clean',]
        self.build_system.options = ['-f Makefile_warmup', f'buildtype={self.build_type}']
        

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    build_type = parameter(['hip', 'acc', 'hipacc', 'omp', 'hipomp'])

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
        _, modules, _ = set_env(mpi = False, gpu = True)
        if modules != []:
            self.modules = modules
        self.prebuild_cmds = [f'make -f Makefile_multigpu clean']
        self.build_system.options = ['-f Makefile_multigpu', f'buildtype={self.build_type}']

    # Test parameter
    build_type = parameter(['hip', 'hipacc', 'acc'])

    # Test passes if compiled executable exists
    @sanity_function
    def assert_compiled(self):
        return sn.assert_true(os.path.exists(f'./multigpu.{self.build_type}.exe'))


# Probably can form a base class, and separate tests for warmup, performance, and offloading tests
@rfm.simple_test
class warmup_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test GPU warmup times using test from performance-modelling-tools repo'
        self.maintainers = ['Craig Meyer']

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name'
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU

        # Job options (these will automatically be set in the job script by ReFrame)
        self.num_tasks = self.ngpus_per_node
        self.num_cpus_per_task = self.cpus_per_gpu
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = False, omp = iomp, gpu = True)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        
        self.keep_files = ['logs/*']
        
        #############################################
        # PERFORMANCE OF DIFFERENT SECTOINS OF CODE #
        #############################################
        # NOTE: Will likely need altering
        self.scaling_factor = self.ngpus_per_node
        if self.build_type in ['hip', 'hipacc']:
            self.reference = {
                '*': {
                    'run_kernel': (2400 * self.scaling_factor**2, None, 0.2),
                    'alloc': (250 * self.scaling_factor, None, 0.2),
                    'free': (200 * self.scaling_factor, None, 0.2),
                    'kernel': (50 * self.scaling_factor, None, 0.2),
                    'd2h': (700 * self.scaling_factor**2, None, 0.2),
                    'h2d': (700 * self.scaling_factor**2, None, 0.2),
                },
            }
        elif 'omp' in self.build_type:
            self.reference = {
                '*': {
                    'run_kernel': (1200 * self.scaling_factor**2, None, 0.2),
                    'alloc': (1000 * self.scaling_factor**2, None, 0.2),
                    'free': (700 * self.scaling_factor**2, None, 0.2),
                    'kernel': (600 * self.scaling_factor**2, None, 0.2),
                    'd2h': (600 * self.scaling_factor**2, None, 0.2),
                    'h2d': (600 * self.scaling_factor**2, None, 0.2),
                }
            }
        elif self.build_type == 'acc':
            self.reference = {
                '*': {
                    'run_kernel': (1100, None, 0.2),
                    'alloc': (600, None, 0.2),
                    'free': (1200, None, 0.2),
                    'kernel': (600, None, 0.2),
                    'd2h': (700, None, 0.2),
                    'h2d': (400, None, 0.2),
                }
            }


        self.tags = {'gpu'}


    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    ngpus_per_node = parameter([2, 8])
    build_type = parameter(['hip', 'hipacc', 'acc', 'omp', 'hipomp'])

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

    ###########################
    # RECOMMENDED JOB OPTIONS #
    ###########################
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [f'--account={self.acct_str}', f'--gres=gpu:{self.ngpus_per_node}']

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

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name'
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU

        # Job options (these will automatically be set in the job script by ReFrame)
        self.num_tasks = self.ngpus_per_node
        self.num_cpus_per_task = self.cpus_per_gpu
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = False, omp = iomp, gpu = True)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        
        self.keep_files = ['logs/*']

        #############################################
        # PERFORMANCE OF DIFFERENT SECTOINS OF CODE #
        #############################################
        # NOTE: Will likely need altering
        self.scaling_factor = self.ngpus_per_node
        if self.build_type in ['hip', 'hipacc']:
            self.reference = {
                '*': {
                    'run_kernel': (8000 * self.scaling_factor**2, None, 0.2),
                    'kernel': (900 * self.scaling_factor**2, None, 0.2),
                    'd2h': (600 * self.scaling_factor**2, None, 0.2),
                    'h2d': (450 * self.scaling_factor**2, None, 0.2),
                },
            }
        elif self.build_type == 'acc':
            self.reference = {
                '*': {
                    'run_kernel': (600, None, 0.2),
                    'kernel': (250, None, 0.2),
                    'd2h': (600, None, 0.2),
                    'h2d': (300, None, 0.2),
                }
            }

        self.tags = {'gpu'}

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    ngpus_per_node = parameter([2, 8])
    build_type = parameter(['hip', 'hipacc', 'acc'])

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

    ###########################
    # RECOMMENDED JOB OPTIONS #
    ###########################
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [f'--account={self.acct_str}', f'--gres=gpu:{self.ngpus_per_node}']

    # Sanity function
    @sanity_function
    def assert_multigpu(self):
        return sn.assert_found('Reporting times for', self.stdout)
