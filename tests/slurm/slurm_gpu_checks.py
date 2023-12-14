# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

# Import functions to set environment variables, etc.
import sys
import os
sys.path.append(os.getcwd() + '/common/scripts')
from set_test_env import *

@rfm.simple_test
class gpu_count_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check access and running of jobs on various GPU partitions'
        self.maintainers = ['Craig Meyer']

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU

        # The executable is meaningless, but needs to be set for test to run
        # Output from postrun_cmds is what is relevant to the test
        self.executable = 'echo'
        self.executable_opts = ['hello world']
        
        # Set command to list all GPUs available to the job
        _, _, cmds = set_env(mpi = False, sched = True, gpu = True)
        if cmds != []:
            self.postrun_cmds = cmds

        # Handle job config for both exclusive and shared access
        if self.access == 'exclusive':
            self.ngpus = self.exclusive_gpus_per_node
            self.exclusive_access = True
        else:
            self.ngpus = 1
        self.num_tasks = 1
        self.time_limit = '2m'
        
        self.tags = {'gpu'}

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    # Check for both exclusive and shared node access
    access = parameter(['exclusive', 'shared'])

    # Job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            '--nodes=1',
            f'--gres=gpu:{self.ngpus}',
            f'--account={self.acct_str}',
        ]

    @sanity_function
    def assert_partition(self):
        # NOTE: regex pattern is for rocm-smi
        # NOTE: if using other command to list GPUs, this may need to be modified
        num_devices = len(sn.evaluate(sn.extractall(r'([0-9]+)-(\w[0-9]+)-([0-9]+)', self.stdout)))
        return sn.assert_eq(num_devices, self.ngpus)


@rfm.simple_test
class gpu_cpu_count_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check the no. of CPUs assigned matches value corresponding to requested no. of GPUs'
        self.maintainers = ['Craig Meyer']

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU
        self.all_gpu_nodes = 'nodelist_str' # String of all nodes in the partition e.g. 'nid[0000, 0001, 0002, ...]'
        self.num_gpu_nodes = len(self.all_gpu_nodes.split(','))

        # The executable is meaningless, but needs to be set for test to run
        # Output from postrun_cmds is what is relevant to the test
        self.executable = 'echo'
        self.executable_opts = ['hello world']
        
        # Set command to list all GPUs available to the job
        _, _, cmds = set_env(mpi = False, sched = True, gpu = True)
        if cmds != []:
            self.postrun_cmds = cmds

        # Handle job config for both exclusive and shared access
        if self.ngpus == self.exclusive_gpus_per_node:
            self.exclusive_access = True
        self.num_tasks = self.ngpus
        self.time_limit = '2m'

        # First one should run successfully, second fail with too many processors
        self.postrun_cmds += [
            f'srun -n {self.num_tasks * 8} -c 1 hostname',
            f'srun -n {self.num_tasks * 8 + 1} - c 1 hostname',
            f'scontrol show nodes {self.all_gpu_nodes} | grep Gres= | sort | uniq -c'
        ]
        
        self.tags = {'gpu'}

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    # Test shared and exclusive access, odd and even no. of GPUs, more and less than half the node
    ngpus = parameter([1, 6, 8])

    # Job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            '--nodes=1',
            f'--gres=gpu:{self.ngpus}',
            f'--account={self.acct_str}',
        ]
        self.job.launcher.options = ['-n 1']

    @sanity_function
    def assert_partition(self):
        # Check no. of charged CPUs, output of hostname, and no. of nodes with "good" Gres parameter
        tres_cpu = sn.extractsingle(r'TRES=cpu=(?P<tres_cpu>\S+),mem.+', self.stdout, 'tres_cpu', int)
        num_hostnames = len(sn.evaluate(sn.extractall(r'^nid.*', self.stdout)))
        num_good_nodes = sn.extractsingle(r'\s+(?P<num_good_nodes>\S+)\s+Gres=gpu:8\(S:0-7\)', self.stdout, 'num_good_nodes', int)

        return sn.all([
            sn.assert_eq(num_good_nodes, self.num_gpu_nodes),
            sn.assert_eq(num_hostnames, self.num_tasks * 8),
            sn.assert_eq(tres_cpu, self.ngpus * 16),
            sn.assert_found('More processors requested than permitted', self.stderr),
        ])

@rfm.simple_test
class gpu_affinity_base_check(rfm.RegressionTest):
    def __init__(self):

        self.descr = 'Base test class for GPU affinity tests'
        self.maintainers = ['Craig Meyer']

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU

        # Compilation - build from Makefile
        self.build_system = 'Make'
        self.build_system.flags_from_environ = False
        self.build_system.options = ['-f Makefile_gpu']
        # Build profile_util library used for code
        self.prebuild_cmds += [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_hip.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}',
        ]
        # Execution
        self.executable = './affinity_report.gpu'
        self.executable_opts = ['| sort -n']

        # Job options - differ between whether running in exclusive or shared access
        if self.access == 'exclusive':
            self.exclusive_access = True
            self.ngpus_per_node = self.exclusive_gpus_per_node
            self.ntasks_per_node = self.exclusive_gpus_per_node
            self.num_tasks = self.exclusive_gpus_per_node
        elif self.access == 'shared':
            self.ngpus_per_node = 2
            self.ntasks_per_node = 2
            self.num_tasks = 2
        self.ngpus_per_task = 1
        self.num_cpus_per_task = self.cpus_per_gpu
        self.num_nodes = 1

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = True, omp = iomp, sched = True, gpu = True)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        
        self.tags = {'gpu'}

    # Test affinity for both exclusive and shared node access
    access = parameter(['exclusive', 'shared'])

    # Set job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--account={self.acct_str}',
            f'--gres=gpu:{self.ngpus_per_node}',
        ]

    # Add CPUs per task option to srun launcher command
    @run_before('run')
    def modify_launcher(self):
        self.job.launcher.options = [
            f'-c {self.ncpus_per_task} --ntasks-per-node={self.ntasks_per_node} --gpus-per-task={self.ngpus_per_task} --gpu-bind=closest'
        ]

    # Test passes if ALL GPUs are pinned to a valid CPU of the correct (closest) L3 cache region
    @sanity_function
    def check_pinning(self):
        # Dictionary of CPUs associated with each GPU on a node
        gpu_dict = {
            'd1': [0,1,2,3,4,5,6,7],
            'd6': [8,9,10,11,12,13,14,15],
            'c9': [16,17,18,19,20,21,22,23],
            'ce': [24,25,26,27,28,29,30,31],
            'd9': [32,33,34,35,36,37,38,39],
            'de': [40,41,42,43,44,45,46,47],
            'c1': [48,49,50,51,52,53,54,55],
            'c6': [56,57,58,59,60,61,62,63],
        }

        # regex pattern of MPI rank -> CPU pinning output
        mpi_cpu = sn.evaluate(
            sn.findall(r'.*On node (nid[0-9]+).*MPI Rank ([0-9]+).*Thread 0.*placement = ([0-9]+)', self.stdout)
        )

        result = True
        for i in range(self.num_tasks):
            cpu_id = int(mpi_cpu[i].groups()[2])
            nid = mpi_cpu[i].groups()[0]
            mpi_rank = int(mpi_cpu[i].groups()[1])
            valid_cpus = []

            # regex pattern of MPI rank -> GPU pinning output
            mpi_gpu = sn.evaluate(
                sn.findall(fr'.*On node {nid}.*MPI Rank {mpi_rank}.*Bus_ID=0000:(\w*[0-9]*):.*', self.stdout)
            )
            for j in range(self.ngpus_per_task):
                gpu_bus_id = mpi_gpu[j].groups()[0]
                valid_cpus += gpu_dict[gpu_bus_id]
                # Check pinned CPU of this MPI rank is a valid CPU for the associated GPU(s)
                if cpu_id not in valid_cpus:
                    result = False

        return sn.assert_true(result)


@rfm.simple_test
class gpu_affinity_array_check(rfm.RegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'GPU affinity test for job arrays'
        self.maintainers = ['Craig Meyer']

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU

        # Compilation - build from Makefile
        self.build_system = 'Make'
        self.build_system.flags_from_environ = False
        self.build_system.options = ['-f Makefile_gpu']
        # Build profile_util library used by code
        self.prebuild_cmds += [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_hip.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}',
        ]
        # Execution
        self.executable = './affinity_report.gpu'
        self.executable_opts = ['| sort -n']

        # Job options
        self.num_nodes = self.job_config[0]
        self.num_array_tasks = self.job_config[1]
        self.ngpus_per_node = self.job_config[2]
        self.ntasks_per_node = self.job_config[3]
        self.num_tasks = self.ntasks_per_node * self.num_nodes
        self.ngpus_per_task = self.ngpus_per_node // self.ntasks_per_node
        self.num_cpus_per_task = self.cpus_per_gpu * self.ngpus_per_task
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = True, omp = iomp, sched = True, gpu = True)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        self.tags = {'gpu'}


    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    # job_config = [`num_nodes`, `num_array_tasks`, `ngpus_per_node`, `ntasks_per_node`]
    # Single-node exclusive and shared access tests (the array fills out an entire node and part of a node)
    # One and multiple GPUs per task tests
    job_config = parameter([
        [1, 4, 2, 2], [1, 2, 3, 3],
        [2, 4, 4, 4], [2, 2, 6, 6],
        [1, 4, 2, 1], [1, 2, 2, 1],
        [2, 4, 4, 2], [2, 2, 2, 1],
    ])

    # Set job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--account={self.acct_str}',
            f'--array=1-{self.num_array_tasks}',
            f'--gres=gpu:{self.ngpus_per_node}',
        ]

    # Add CPUs per task option to srun launcher command
    @run_before('run')
    def modify_launcher(self):
        self.job.launcher.options = [
            f'-c {self.ncpus_per_task} --ntasks-per-node={self.ntasks_per_node} --gpus-per-task={self.ngpus_per_task} --gpu-bind=closest'
        ]

    # Test passes if ALL GPUs are pinned to a valid CPU of the correct (closest) L3 cache region
    @sanity_function
    def check_pinning(self):
        # Dictionary of CPUs in correct L3 cache for each GPU
        gpu_dict = {
            'd1': [0,1,2,3,4,5,6,7],
            'd6': [8,9,10,11,12,13,14,15],
            'c9': [16,17,18,19,20,21,22,23],
            'ce': [24,25,26,27,28,29,30,31],
            'd9': [32,33,34,35,36,37,38,39],
            'de': [40,41,42,43,44,45,46,47],
            'c1': [48,49,50,51,52,53,54,55],
            'c6': [56,57,58,59,60,61,62,63],
        }

        # regex pattern of MPI rank -> CPU pinning output
        mpi_cpu = sn.evaluate(
            sn.findall(r'.*On node (nid[0-9]+).*MPI Rank ([0-9]+).*Thread 0.*placement = ([0-9]+)', self.stdout)
        )
        # regex pattern of MPI rank -> GPU pinning output
        mpi_gpu = sn.evaluate(sn.findall(r'.*On node (nid[0-9]+).*MPI Rank ([0-9]+).*Bus_ID=0000:(\w*[0-9]*):.*', self.stdout))
        result = True

        idx = 0
        for i in range(0, self.num_array_tasks * self.num_tasks):
            cpu_id = int(mpi_cpu[i].groups()[2])
            mpi_rank = int(mpi_cpu[i].groups()[1])
            valid_cpus = []

            for j in range(self.ngpus_per_task):
                gpu_bus_id = mpi_gpu[idx + j].groups()[2]
                valid_cpus += gpu_dict[gpu_bus_id]
            idx += self.ngpus_per_task
            # Check pinned CPU of this MPI rank is a valid CPU for the associated GPU(s)
            if cpu_id not in valid_cpus:
                result = False
        
        return sn.assert_true(result)


@rfm.simple_test
class gpu_affinity_jobpacking_check(rfm.RegressionTest):
    def __init__(self):

        # Metadta
        self.descr = 'GPU affinity test for job packing (job steps)'
        self.maintainers = ['Craig Meyer']

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.acct_str = 'account_name' # Account to charge job to
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU
        self.mem_per_gpu = 29440 # Memory per GPU in MB

        # Compilation
        self.build_system = 'Make'
        self.build_system.flags_from_environ = False
        self.build_system.options = ['-f Makefile_gpu']
        # Build profile_util library used by code
        self.prebuild_cmds += [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_hip.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}',
        ]
        # Execution
        self.executable = './affinity_report.gpu'
        self.executable_opts = ['| sort -n &']

        # Job options
        self.num_nodes = self.job_config[0]
        self.ngpus_per_node_1 = self.job_config[1]
        self.ngpus_per_node_2 = self.job_config[2]
        self.ngpus_per_task = self.job_config[3]
        self.ngpus_per_node = self.ngpus_per_node_1 + self.ngpus_per_node_2
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True
        self.ntasks_per_node_1 = self.ngpus_per_node_1 // self.ngpus_per_task
        self.ntasks_per_node_2 = self.ngpus_per_node_2 // self.ngpus_per_task
        self.ntasks_per_node = self.num_gpus_per_node // self.ngpus_per_task
        self.ntasks_1 = self.ntasks_per_node_1 * self.num_nodes
        self.ntasks_2 = self.ntasks_per_node_2 * self.num_nodes
        self.num_tasks = self.ntasks_1 + self.ntasks_2
        self.num_cpus_per_task = self.cpus_per_gpu * self.ngpus_per_task
        self.mem_1 = self.ngpus_per_node_1 * self.mem_per_gpu
        self.mem_2 = self.ngpus_per_node_2 * self.mem_per_gpu

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = True, omp = iomp, sched = True, gpu = True)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
            
       self.tags = {'gpu'}
            
    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    # Test parameters - job_config = [`num_nodes`, `ngpus_per_node` (for each of the two jobsteps), `ngpus_per_task`]
    job_config = parameter([
        [1, 3, 1, 1], [1, 5, 3, 1],
        [2, 3, 1, 1], [2, 5, 3, 1],
        [1, 4, 2, 2], [1, 6, 2, 2],
        [2, 4, 2, 2], [2, 6, 2, 2],
    ])

    # Set job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--account={self.acct_str}',
            f'--gres=gpu:{self.ngpus_per_node}',
        ]

    # Add CPUs per task option to srun launcher command
    @run_before('run')
    def modify_launcher(self):
        self.job.launcher.options = [
            '--exact',
            f'-N {self.num_nodes} -n {self.ntasks_1} --ntasks-per-node={self.ntasks_per_node_1} -c {self.ncpus_per_task}',
            f'--gpus-per-node={self.ngpus_per_node_1} --gpus-per-task={self.ngpus_per_task} --gpu-bind=closest',
        ]
    @run_before('run')
    def run_second_jobstep(self):
        self.postrun_cmds += [
            f'srun --exact ' +
            f'-N {self.num_nodes} -n {self.ntasks_2} --ntasks-per-node={self.ntasks_per_node_2} -c {self.ncpus_per_task} ' +
            f'--gpus-per-node={self.ngpus_per_node_2} --gpus-per-task={self.ngpus_per_task} --gpu-bind=closest ' +
            f'{self.executable} | sort -n &',
            'wait'
        ]

    # Test passes if ALL GPUs are pinned to a valid CPU of the correct (closest) L3 cache region
    @sanity_function
    def check_pinning(self):
        # Dictionary of CPUs in correct L3 cache for each GPU
        gpu_dict = {
            'd1': [0,1,2,3,4,5,6,7],
            'd6': [8,9,10,11,12,13,14,15],
            'c9': [16,17,18,19,20,21,22,23],
            'ce': [24,25,26,27,28,29,30,31],
            'd9': [32,33,34,35,36,37,38,39],
            'de': [40,41,42,43,44,45,46,47],
            'c1': [48,49,50,51,52,53,54,55],
            'c6': [56,57,58,59,60,61,62,63],
        }

        # regex pattern of MPI rank -> CPU pinning output
        mpi_cpu = sn.evaluate(sn.findall(r'.*MPI Rank ([0-9]+).*Thread 0.*placement = ([0-9]+)', self.stdout))
        # regex pattern of MPI rank -> GPU pinning output
        mpi_gpu = sn.evaluate(sn.findall(fr'.*MPI Rank ([0-9]+).*Bus_ID=0000:(\w*[0-9]*):.*', self.stdout))
        result = True
        
        idx = 0
        for i in range(self.num_tasks):
            cpu_id = int(mpi_cpu[i].groups()[1])
            valid_cpus = []

            mpi_rank = int(mpi_gpu[i].groups()[0])
            for j in range(self.ngpus_per_task):
                gpu_bus_id = mpi_gpu[idx + j].groups()[1]
                valid_cpus += gpu_dict[gpu_bus_id]
            idx += self.ngpus_per_task
            # Check pinned CPU of this MPI rank is a valid CPU for the associated GPU(s)
            if cpu_id not in valid_cpus:
                result = False
        
        return sn.assert_true(result)
