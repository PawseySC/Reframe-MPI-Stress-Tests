# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn



# Count the occurencess of assembly code instructions in assembly file
@rfm.simple_test
class countInstructions(rfm.CompileOnlyRegressionTest):
    def __init__(self):

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        # Valid systems and programming environments
        self.valid_systems = ['system:login']
        self.valid_prog_environs = ['*']

        # Metadata
        self.descr = 'Test class for counting instructions in assembly file'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']


        # Setup depends on what system we are on
        self.sysname = self.current_system.name

        # Setup for compile-time
        # We build `sourcepath` into an assembly-only "executable" `executable`
        self.build_system = 'SingleSource'
        self.sourcepath = 'vec_calc.cpp'
        self.executable = 'assembly.s'
        self.cppflags = {
                'system': [f'-{self.oflag}', f'-march={self.arch}'],
            }

        # Keep assembly files produced from compilaton so can be analysed
        # by python script in the postbuild_cmds
        self.keep_files = [self.executable]

        self.reference = {
            'system': {self.target_str: (1, 0, None, 'counts')}
        }
        self.perf_variables = {self.target_str: self.count_instructions(self.target_str)}

        # If a common instruction of interest is found to be represented 
        # differently in assembly language from different compilers, 
        # construct a lookup dictionary here
    

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    target_str = parameter(['fma']) # Instruction representation in assembly code
    oflag = parameter(['O0', 'O1', 'O2', 'O3'])
    arch = parameter(['znver2', 'znver3'])
    ompflags = parameter(['', '-fopenmp'])
    mpiflags = parameter(['', '-D_MPI'])

    # Set the compile flags that are the same for each child test
    @run_before('compile')
    def set_cppflags(self):
        self.build_system.cppflags += ['-S'] # Always produce an assembly file
        self.build_system.cppflags += [self.ompflags]
        self.build_system.cppflags += [self.mpiflags]
        self.build_system.cppflags += self.cppflags[self.sysname]
    # @run_before('performance')
    # def set_perf_dict(self):
    #     self.perv_variables = {instr: self.count_instructions(instr)}
    @performance_function
    def count_instructions(self, instruction = 'fma'):
        return len(sn.evaluate(sn.extractall(self.target_str, self.executable)))

    # Sanity test - fail if no instances of `target_str` found
    @sanity_function
    def assert_instr(self):
        num_instrs = len(sn.evaluate(sn.extractall(self.target_str, self.executable)))
        return sn.assert_ge(num_instrs, 1)