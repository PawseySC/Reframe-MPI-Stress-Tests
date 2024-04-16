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


@rfm.simple_test
class benchmarkInstructions(rfm.RunOnlyRegressionTest):
    def __init__(self):

        sys_info = set_system(config_path, 'benchmarkInstructions')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        job_info = get_job_options(config_path, 'benchmarkInstructions')
        self.acct_str = job_info['account']

        # Metadata
        self.descr = 'Test class for analysing compiler assembly code and impact on code performance'
        self.maintainers = ['Craig', 'Pascal Jahan Elahi']

        # Setup depends on what system we are on
        self.sysname = self.current_system.name
        test_config = configure_test(config_path, 'benchmarkInstructions')
        # Explicitly specify the number of CPUs per task
        self.num_cpus_per_task = 1

        self.sourcepath = 'vec_bm.cpp'
        self.cppflags = [
            self.ompflag, self.mpiflag, self.oflag, self.arch,
            '-L${PROFILE_UTIL_DIR}/lib', '-I${PROFILE_UTIL_DIR}/include', '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/',
            self.sourcepath] # -o vec_bm.out
        self.executable = 'vec_bm.out'
        self.prerun_cmds = [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_cpu.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}',
        ]

        self.keep_files = ['*.s']

        self.ref_val = test_config['performance']['reference-value']
        self.reference = {
            self.sysname: {self.target_str: (self.ref_val, 0, None, 'counts')}
        }

    # Test parameters
    params = get_test_params(config_path, 'benchmarkInstructions')
    target_str = parameter(params['instruction-string']) # Instruction representation in assembly code
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
                self.cppflags += ['-lprofile_util']
            else:
                self.cppflags += ['-lprofile_util_mpi']
        else:
            if self.mpiflag == '':
                self.cppflags += ['-lprofile_util_omp']
            else:
                self.cppflags += ['-lprofile_util_mpi_omp']
        self.cppflags += ['-S']#, '-o vec_bm.out']
        # Compilation to get assembly file
        self.prerun_cmds += ['CC' + ' '.join(self.cppflags)]
        # Compilation to get executable
        self.prerun_cmds += ['CC' + ' '.join(self.cppflags[:-1] + [f'-o {self.executable}'])]
    # Explicitly set `-c` in srun statement
    @run_before('run')
    def srun_cpus_per_task(self):
        self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
    # Benchmark the allocation of memory, initialisation of vectors,
    # and the vector math operations
    @run_before('performance')
    def set_perf_dict(self):
        self.perf_variables = {
            self.target_str: self.count_instructions(),
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
    def count_instructions(self):
        return len(sn.evaluate(sn.extractall(self.target_str, 'vec_bm.s')))

    # Simple sanity function to see if program reaches completion
    @sanity_function
    def assert_vec_calc(self):
        return sn.assert_found('Vector calculation finished', self.stdout)