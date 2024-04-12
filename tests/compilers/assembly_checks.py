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



# Count the occurencess of assembly code instructions in assembly file
@rfm.simple_test
class countInstructions(rfm.CompileOnlyRegressionTest):
    def __init__(self):

        sys_info = set_system(config_path, 'countInstructions')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Metadata
        self.descr = 'Test class for counting instructions in assembly file'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        # Setup depends on what system we are on
        self.sysname = self.current_system.name
        test_config = configure_test(config_path, 'countInstructions')

        # Setup for compile-time
        # We build `sourcepath` into an assembly-only "executable" `executable`
        self.build_system = 'SingleSource'
        self.sourcepath = 'vec_calc.cpp'
        self.executable = 'assembly.s'
        self.cppflags = {
                self.sysname: [f'-{self.oflag}', f'-march={self.arch}'],
            }

        # Keep assembly files produced from compilaton
        self.keep_files = [self.executable]

        self.ref_val = test_config['performance']['reference-value']
        self.reference = {
            self.sysname: {self.target_str: (self.ref_val, 0, None, 'counts')}
        }
        self.perf_variables = {self.target_str: self.count_instructions()}
    

    # Test parameters
    params = get_test_params(config_path, 'countInstructions')
    target_str = parameter(params['instruction-string']) # Instruction representation in assembly code
    oflag = parameter(params['oflag'])
    arch = parameter(params['arch'])
    ompflags = parameter(params['ompflags'])
    mpiflags = parameter(params['mpiflags'])

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
    @performance_function('counts')
    def count_instructions(self):
        return len(sn.evaluate(sn.extractall(self.target_str, self.executable)))

    # Sanity test - fail if no instances of `target_str` found
    @sanity_function
    def assert_instr(self):
        num_instrs = len(sn.evaluate(sn.extractall(self.target_str, self.executable)))
        return sn.assert_ge(num_instrs, 1)