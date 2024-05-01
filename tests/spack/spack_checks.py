# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

import re
import os
import sys
import yaml
import json

# Add root directory of repo to path
curr_dir = os.path.dirname(__file__).replace('\\','/')
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)
from tests.spack.spack_helper_methods import *
# Import functions to set env vars, modules, commands
from common.scripts.parse_yaml import *
config_path = curr_dir + '/spack_config.yaml'


# Test to check that every abstract spec in spack.yaml file 
# has a  matching concretised spec in spack.lock file
@rfm.simple_test
class concretise_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check that every spec in an environment was concretised successfully'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Execution
        self.executable = 'echo'
        self.executable_opts = ['Checking conretisation success']

        self.tags = {'spack', 'concretization', 'software_stack'}

    
    @sanity_function
    def assert_concretisation(self):
        abstract_specs = get_abstract_specs()
        root_specs = get_root_specs()
        abstract_name_ver = [None] * len(abstract_specs)
        root_name_ver = [None] * len(root_specs)

        # All abstract specs in {name}/{version} format
        abstract_pattern = r'([\w-]+@*=*[\w.]+).*'
        idx = 0
        for s in abstract_specs:
            match = re.match(abstract_pattern, s)
            if match != None:
                if '@' in match.groups()[0]:
                    abstract_name_ver[idx] = match.groups()[0].split('@')[0] + '/' + match.groups()[0].split('@')[1].replace('=', '')
                else:
                    abstract_name_ver[idx] = match.groups()[0]
            idx += 1

        # All concretised specs in {name}/{version} format
        concrete_pattern = r'^([\w-]+@*=*[\w.]+)'
        idx = 0
        for s in root_specs:
            match = re.match(concrete_pattern, s).groups()[0]
            if '@' in match:
                root_name_ver[idx] = match.split('@')[0] + '/' + match.split('@')[1].replace('=', '')
            else:
                root_name_ver[idx] = match
            idx += 1

        # Check if an abstract spec is missing from the list of concretised specs
        num_failed = 0
        for spec in abstract_name_ver:
            # True if empty list (i.e. there is no match for `spec` found)
            if not [m for m in root_name_ver if spec in m]:
                num_failed += 1

        # Test only passes if all abstract specs have a matching concretised spec
        return sn.assert_lt(num_failed, 1)

# Test to check that a module and all its dependent modules exist
@rfm.simple_test
class module_existence_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check for existence of a module during software stack installation'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Execution - ls to check the module exists
        self.prerun_cmds = ['ls -R .', 'echo $(pwd)']
        self.executable = 'ls'
        self.executable_opts = [self.mod]
        # Get dependencies for the module and add ls commands for those
        dependencies = get_module_dependencies(self.mod)
        if len(dependencies) > 0:
            self.postrun_cmds = [f'ls {d}' for d in dependencies]

        self.tags = {'spack', 'installation', 'software_stack'}

    # Test parameter - list of full absolute paths for every module in the environment
    mod = parameter(get_module_paths())

    @sanity_function
    def assert_module_exists(self):
        return sn.assert_not_found('No such file or directory', self.stderr)


# Test to check that a module and all of its load dependencies (if any) load correctly
@rfm.simple_test
class module_load_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check a module can load properly during software stack installation'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Execution
        self.executable = 'module'
        self.name_ver = '/'.join(self.mod.split('/')[-2:])[:-4]
        self.executable_opts = ['load', self.name_ver]

        # module show to check the correct module is being pointed to
        self.prerun_cmds += [f'module show {self.name_ver}']
        # Check main module is loaded 
        self.postrun_cmds = [f'if module is-loaded {self.name_ver} ; then echo "main package is loaded"; fi']

        self.tags = {'spack', 'installation', 'software_stack'}
    
    # Test parameter - list of full absolute paths for every module in the environment
    mod = parameter(get_module_paths())
    
    # Dependency - this test only runs if the corresponding `module_existence_check` test passes
    # Dependency - the test dependency depends on the environment
    @run_after('init')
    def inject_dependencies(self):
        testdep_name = f'module_existence_check_{self.mod}'
        # ReFrame replaces instances of "/", ".", "-", and "+" in test name with "_"
        chars = "/.-+"
        for c in chars:
            testdep_name = testdep_name.replace(c, '_')
        self.depends_on(testdep_name, udeps.by_env)
    # Add commands to check that dependent modules are loaded
    @run_before('run')
    def check_load_lines(self):
        # Get list of dependencies that need to be loaded - explicit load statements in module file
        self.load_lines = [line.split('load(')[-1][:-2].replace('"', '') for line in open(self.mod).readlines() if line.startswith('load')]
        nloads = len(self.load_lines)
        # `++` breaks the regex search, so replace ++ with \+\+ if present
        for i in range(nloads):
            if '++' in self.load_lines[i]:
                l = self.load_lines[i]
                self.load_lines[i] = l.replace('++', '\+\+')
        # Check all dependencies are loaded
        self.postrun_cmds += [f'if module is-loaded {dep_mod} ; then echo "dependency is loaded"; fi' for dep_mod in self.load_lines]

    @sanity_function
    def assert_module_loaded(self):
        # ++, if present, breaks regex search
        if '++' in self.mod:
            self.mod = self.mod.replace('++', '\+\+')
        if '++' in self.name_ver:
            self.name_ver = self.name_ver.replace('++', '\+\+')
        
        # Test passes if main module is loaded, all dependent packages are loaded,
        # `module show` points to correct module, and no failures/errors are found
        return sn.all([
            sn.assert_found("main package is loaded", self.stdout),
            sn.assert_eq(sn.count(sn.extractall('dependency is loaded', self.stdout)), len(self.load_lines)),
            sn.assert_found(self.mod, self.stderr),
            sn.assert_not_found('Failed', self.stderr),
            sn.assert_not_found('Error', self.stderr),
        ])


# Run the most basic sanity check for a loaded modue
# For software - `--help` or `--version`
# For library - `ldd`
@rfm.simple_test
class baseline_sanity_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check that, once the module is loaded, the software shows the most minimal functionality (--help or --version)'
        self.amintainers = ['Craig Meyer']

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Load the module we are testing
        self.name_ver = '/'.join(self.mod.split('/')[-2:])[:-4]
        self.modules = [self.name_ver]

        # Get the base package name from full module path
        self.base_name = self.mod.split('/')[-2]
        # Use dictionary to set executable
        self.pkg_cmds = get_pkg_cmds(curr_dir + '/pkg_cmds.yaml')
        self.executable = self.pkg_cmds[self.base_name][0]
        # Set the executable options, which depends on if it's software or library
        if self.executable == 'ldd':
            sw_path = get_software_path(self.mod.split('/')[-2:])
            self.executable_opts = [sw_path + '/' + self.pkg_cmds[self.base_name][1]]
        else:
            # Pipe stdout and stderr to stdout
            self.executable_opts = [self.pkg_cmds[self.base_name][1] + ' 2>&1']
        
        self.tags = {'spack', 'installation', 'software_stack'}
    
    # Test parameter - list of full absolute paths for every module in the environment
    mod = parameter(get_module_paths())

    # Dependency - this test only runs if the corresponding `module_load_check` test passes
    # Dependency - the test dependency depends on the environment
    @run_after('init')
    def inject_dependencies(self):
        testdep_name = f'module_load_check_{self.mod}'
        # ReFrame replaces instances of "/" and "." in test name with "_"
        chars = "/.-+"
        for c in chars:
            testdep_name = testdep_name.replace(c, '_')
        self.depends_on(testdep_name, udeps.by_env)


    @sanity_function
    def assert_functioning(self):
        # For libraries we check if any of the libraries return from `ldd` are not found
        if self.executable == 'ldd':
            return sn.assert_not_found('not found', self.stdout)
        # For software we do a basic check (e.g. --help or --version)
        else:
            return sn.assert_found(self.pkg_cmds[self.base_name][2], self.stdout)
