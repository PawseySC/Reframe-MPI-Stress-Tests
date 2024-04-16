# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

import sys
import os.path

# Add root directory of repo to path
curr_dir = os.path.dirname(__file__).replace('\\','/')
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)
# Import functions to set env vars, modules, commands
from common.scripts.parse_yaml import *
config_path = curr_dir + '/cpu_compilation_config.yaml'


# Test for compiling with given optimisations and benchmarking performance
# metrics of a vector calculation code
@rfm.simple_test
class benchmarkOptimisations(rfm.RegressionTest):
    def __init__(self):

        sys_info = set_system(config_path, 'benchmarkOptimisations')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        job_info = get_job_options(config_path, 'benchmarkOptimisations')
        self.acct_str = job_info['account']

        # Metadata
        self.descr = 'Test class for analysing compiler optimisations and impact on code performance'
        self.maintainers = ['Craig', 'Pascal Jahan Elahi']

        # Setup depends on what system we are on
        self.sysname = self.current_system.name
        test_config = configure_test(config_path, 'benchmarkOptimisations')
        self.ofile_flag = test_config['system-parameters']['ofile-flag']
        # Explicitly specify the number of CPUs per task
        self.num_cpus_per_task = 1

        # Setup for compile-time
        # We build `sourcepath`, creating an optimisation file `self.ofile`
        self.build_system = 'SingleSource'
        self.sourcepath = 'vec_bm.cpp'
        self.ofile = 'vec_bm.lst'
        self.build_system.cppflags = [
            '-L${PROFILE_UTIL_DIR}/lib', 
            '-I${PROFILE_UTIL_DIR}/include', 
            '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/', 
            ]
        self.build_system.cppflags += [self.ompflag, self.mpiflag]
        self.build_system.cppflags += [self.oflag, self.arch]
        self.build_system.cppflags += [f'{self.ofile_flag}={self.ofile}']
        self.prebuild_cmds += [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_cpu.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}'
        ]

        # Keep optimisation file produced from compilation
        self.keep_files = [f'{self.ofile}']

        self.ref_val = test_config['performance']['reference-value']
        self.reference = {
            self.sysname: {self.opt_string: (self.ref_val, 0, None, 'counts')}
        }

    # Test parameters
    params = get_test_params(config_path, 'benchmarkOptimisations')
    opt_string = parameter(params['optimisation-string'])
    oflag = parameter(params['oflag'])
    arch = parameter(params['arch'])
    ompflag = parameter(params['ompflags'])
    mpiflag = parameter(params['mpiflags'])

    # Set job account in sbatch script
    @run_before('compile')
    def set_account(self):
        self.job.options = [f'--account={self.acct_str}']
    # Set the compilation flags
    @run_before('compile')
    def set_cppflags(self):
        # We want to compare with/without OMP/MPI, so set profile_util
        # library depending on value of ompflags and mpiflags parameters
        if self.ompflag == '':
            if self.mpiflag == '':
                self.build_system.cppflags += ['-lprofile_util']
            else:
                self.build_system.cppflags += ['-lprofile_util_mpi']
        else:
            if self.mpiflag == '':
                self.build_system.cppflags += ['-lprofile_util_omp']
            else:
                self.build_system.cppflags += ['-lprofile_util_mpi_omp']
    # Explicitly set `-c` in srun statement
    @run_before('run')
    def srun_cpus_per_task(self):
        self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
    # Benchmark the allocation of memory, initialisation of vectors,
    # and the vector math operations
    @run_before('performance')
    def set_perf_dict(self):
        self.perf_variables = {
            self.opt_string: self.count_opts(),
            'Allocate memory': self.extract_timing('allocateMem'),
            'Initialise vectors': self.extract_timing('initialiseVecs'),
            'Perform vector math': self.extract_timing(),
        }

    # Performance functions to measure different 
    # stages of vector calculation code
    @performance_function('us')
    def extract_timing(self, func='doVecMath'):
        if func not in ('allocateMem', 'initialiseVecs', 'doVecMath'):
            raise ValueError(f'Illegal value in argument func ({func!r})')
        
        # Extract timing for the specific function
        return sn.extractsingle(rf'Time taken between : @{func} L[0-9]+ - @{func} L[0-9]+ :\s+(\S+)\s+',
                                self.stdout, 1, float)
    @performance_function('counts')
    def count_opts(self):
        num_opts = len(sn.evaluate(sn.extractall(self.opt_string, self.ofile)))

        return num_opts
    
    # Simple sanity function to see if program reaches completion
    @sanity_function
    def assert_vec_calc(self):
        return sn.assert_found('Vector calculation finished', self.stdout)