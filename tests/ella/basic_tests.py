# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

# Check a test on a login node works
@rfm.simple_test
class countInstructions(rfm.CompileOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test class for counting instructions in assembly file'
        self.maintainers = ['Craig', 'Pascal Jahan Elahi']

        # Valid systems and programming environments
        self.valid_systems = ['ella:login']
        self.valid_prog_environs = ['PrgEnv-gnu']

        # Setup depends on what system we are on
        self.sysname = self.current_system.name

        # Setup for compile-time
        # We build `sourcepath` into an assembly-only "executable" `executable`
        self.build_system = 'SingleSource'
        self.sourcepath = 'vec_calc.cpp'
        self.executable = 'assembly.s'
        self.cppflags = {
                'ella': [f'-{self.oflag}', f'-march={self.arch}'],
            }

        # Keep assembly files produced from compilaton so can be analysed
        # by python script in the postbuild_cmds
        self.keep_files = [f'{self.executable}']

        # If a common instruction of interest is found to be represented 
        # differently in assembly language from different compilers, 
        # construct a lookup dictionary here, similar to what is in `OptimiseBase.py`
    
    # Parameterise the test with nultiple/different optimisation levels,
    # architectures, and target instruction strings (in the assembly code)
    target_str = parameter(['fma'])
    oflag = parameter(['O3'])
    arch = parameter(['native'])
    ompflags = parameter([''])
    mpiflags = parameter([''])

    # Set the compile flags that are the same for each child test
    @run_before('compile')
    def set_cppflags(self):
        self.build_system.cppflags += ['-S'] # Always produce an assembly file
        self.build_system.cppflags += [self.ompflags]
        self.build_system.cppflags += [self.mpiflags]
        self.build_system.cppflags += self.cppflags[self.current_system.name]

    # Sanity test - fail if no instances of `target_str` found
    @sanity_function
    def assert_instr(self):
        num_instrs = len(sn.evaluate(sn.extractall(self.target_str, self.executable)))
        if num_instrs > 0:
            print('There are %d instances of `%s` in this assembly file' % (num_instrs, self.target_str))
        else:
            print('There are NO instances of `%s` in this assembly file' % self.target_str)
        return sn.assert_ge(num_instrs, 1)


# Check that rfm + slurm integration works for a single-node job
@rfm.simple_test
class ella_singlenode_hostname_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Check hostname of single-node job for most basic slurm + Reframe integration'
        self.maintainers = ['Craig']

        # Valid systems and PEs
        self.valid_systems = ['ella:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']

        # Execution
        self.executable = 'hostname'

        # Job config
        self.hostnode = 'ella-n008'
        self.num_tasks_per_node = 4
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_cpus_per_task = 4

        self.prerun_cmds = ['nvidia-smi']

    @run_before('run')
    def set_job_opts(self):
        self.job.options = [f'-w {self.hostnode}', f'--nodes=1']

    @sanity_function
    def assert_hostname(self):

        return sn.assert_found(self.hostnode, self.stdout)

# Check that rfm + slurm integration works for a multi-node job
@rfm.simple_test
class ella_multinode_hostname_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Check hostname of multi-node jof for basic slurm + Reframe integration'
        self.maintainers = ['Craig']

        # Valid systems and PEs
        self.valid_systems = ['ella:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']

        # Execution
        self.executable = 'hostname'

        # Job config
        self.hostnode = 'ella-n007,ella-n008'
        self.num_tasks_per_node = 4
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_cpus_per_task = 4

        self.prerun_cmds = ['nvidia-smi']

    @run_before('run')
    def set_job_opts(self):
        self.job.options = [f'-w {self.hostnode}', '--nodes=2']

    @sanity_function
    def assert_hostname(self):

        return sn.assert_found(self.hostnode, self.stdout)
