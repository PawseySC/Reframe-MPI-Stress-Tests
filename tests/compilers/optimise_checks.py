# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

# Base test class
class OptimiseBase(rfm.CompileOnlyRegressionTest, pin_prefix = True):
    def __init__(self, name, **kwargs):

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:work']
        self.valid_prog_environs = ['*']

        # Metadata
        self.descr = 'Base test class for analysing compiler optimisations'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        # Setup depends on what system we are on
        self.sysname = self.current_system.name

        # Setup for compile-time
        # We build `sourcepath`, creating an optimisation file `self.ofile`
        self.build_system = 'SingleSource'
        self.sourcepath = 'vec_calc.cpp'
        self.ofile = 'vec_calc.lst'

        ##########################################################
        # MAY NEED TO BE EDITED DEPENDING ON COMPILER BEING USED #
        ##########################################################
        self.cppflags = {
                'setonix': [f'-{self.oflag}', f'-march={self.arch}'],
            }

        # Keep optimisation file produced from compilation
        self.keep_files = [f'{self.ofile}']

        # Compilation flags - includes flags to save optimisation file
        # which differ depending on the compiler/PE being used
        self.iflags = {}
        for env in self.valid_prog_environs:
            if 'gnu' in env:
                self.iflags[env] = f'-fopt-info-all={self.ofile}'
            #elif ('cray' in env) or ('aocc' in env):
            else:
                self.iflags[env] = f'-foptimization-record-file={self.ofile}'


        # Dictionary of optimisations and their corresponding strings
        # based on which compiler is being used
        self.opt_strings = {
            'Vectorise': {},
            'Unroll': {},
        }
        for env in self.valid_prog_environs:
            if 'gnu' in env:
                self.opt_strings['Vectorise'][env] = 'loop vectorized'
                self.opt_strings['Unroll'][env] = 'unroll'
            #elif ('cray' in env) or ('aocc' in env):
            else:
                self.opt_strings['Vectorise'][env] = 'vectorized loop'
                self.opt_strings['Unroll'][env] = 'unrolled loop'

    
    # Parameterise the test with nultiple/different optimisation levels,
    # architectures, and target instruction strings (in the assembly code)
    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    oflag = parameter()
    arch = parameter()
    ompflags = parameter(['', '-fopenmp'])
    mpiflags = parameter(['', '-D_MPI'])

    # Set the compile flags that are the same for each child test
    @run_before('compile')
    def set_cppflags(self):
        self.build_system.cppflags = [self.iflags[self.current_environ.name]]
        self.build_system.cppflags += [self.ompflags]
        self.build_system.cppflags += [self.mpiflags]
        if self.sysname == 'mulan':
            self.prebuild_cmds = self.source_cmds[self.curent_environ.name]
        self.build_system.cppflags += self.cppflags[self.current_system.name]
        
# Test for tallying the total number of a given optimisation in an optimisation file
@rfm.simple_test
class countOptimisations(OptimiseBase):
    def __init__(self, **kwargs):
        super().__init__('countOptimisations', **kwargs)

        # # Performance reference values dictionary
        # self.reference = {
        #     'system:work': {'Vectorise': (1 -0.05, None. 'counts')},
        #     'system:work': {'Unroll': (1, -0.05, None, 'counts')}
        # }

    # Parameterise the test with nultiple/different optimisation levels,
    # architectures, and target instruction strings (in the assembly code)
    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    oflag = parameter(['O0', 'O1', 'O2', 'O3'])
    arch = parameter(['x86-64', 'znver3'])
    opts = parameter(['Vectorise', 'Unroll'])

    # @run_before('performance')
    # def set_perf_variables(self):
    #     self.perf_var
    @performance_function('counts')
    def count_opts(self):
        target_str = self.opt_strings[self.opts][self.current_environ.name]
        num_opts = len(sn.evaluate(sn.extractall(target_str, self.ofile)))

        return num_opts


    # Sanity test - fail if no instances of `target_str` found
    @sanity_function
    def assert_opt(self):
        target_str = self.opt_strings[self.opts][self.current_environ.name]
        num_opts = len(sn.evaluate(sn.extractall(target_str, self.ofile)))
        # if num_opts > 0:
        #     print('There are %d instances of `%s` in this optimisation file' % (num_opts, target_str))
        # else:
        #     print('There are NO instances of `%s` in this optimisation file' % target_str)

        return(sn.assert_ge(num_opts, 1))


# Test for compiling with given optimisations and benchmarking performance
# metrics of a vector calculation code
@rfm.simple_test
class benchmarkInstructions(rfm.RegressionTest):
    def __init__(self):

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:work']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to

        # Metadata
        self.descr = 'Test class for analysing compiler optimisations and impact on code performance'
        self.maintainers = ['Craig', 'Pascal Jahan Elahi']

        # Setup depends on what system we are on
        self.sysname = self.current_system.name
        # Explicitly specify the number of CPUs per task
        self.num_cpus_per_task = 1

        # Setup for compile-time
        # We build `sourcepath`, creating an optimisation file `self.ofile`
        self.build_system = 'SingleSource'
        self.sourcepath = 'vec_bm.cpp'
        self.build_system.cppflags = [
            '-L${PROFILE_UTIL_DIR}/lib', 
            '-I${PROFILE_UTIL_DIR}/include', 
            '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/', 
            ]
        ##########################################################
        # MAY NEED TO BE EDITED DEPENDING ON COMPILER BEING USED #
        ##########################################################
        self.cppflags = {
                'system': [f'-{self.oflag}', f'-march={self.arch}'],
            }

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    opts = parameter(['Vectorise'])
    oflag = parameter(['O3'])
    arch = parameter(['znver3'])
    ompflags = parameter(['', '-fopenmp'])
    mpiflags = parameter(['', '-D_MPI'])

    # Set job account in sbatch script
    @run_before('compile')
    def set_account(self):
        self.job.options = [f'--account={self.acct_str}']
    # Compile `profile_util`
    @run_before('compile')
    def compile_prof_util(self):
        self.prebuild_cmds += [
            'cd profile_util', './build_cpu.sh', 'PROFILE_UTIL_DIR=$(pwd)', 'cd ../',
        ]
    # Set the compilation flags
    @run_before('compile')
    def set_cppflags(self):
        # Add the parameters as compilation flags
        self.build_system.cppflags += [self.ompflags]
        self.build_system.cppflags += [self.mpiflags]
        self.build_system.cppflags += self.cpp_flags[self.current_system.name]
        # We want to compare with/without OMP/MPI, so set profile_util
        # library depending on value of ompflags and mpiflags parameters
        if self.ompflags == '':
            if self.mpiflags == '':
                self.build_system.cppflags += ['-lprofile_util']
            else:
                self.build_system.cppflags += ['-lprofile_util_mpi']
        else:
            if self.mpiflags == '':
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
    
    # Simple sanity function to see if program reaches completion
    @sanity_function
    def assert_vec_calc(self):
        return sn.assert_found('Vector calculation finished', self.stdout)