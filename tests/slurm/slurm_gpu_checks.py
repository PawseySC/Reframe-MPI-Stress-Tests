# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from math import floor

import sys
import os.path

# Add root directory of repo to path
curr_dir = os.path.dirname(__file__).replace('\\','/')
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)
# Import functions to set env vars, modules, commands
from common.scripts.parse_yaml import *
config_path = curr_dir + '/slurm_gpu_config.yaml'

@rfm.simple_test
class gpu_count_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check access and running of jobs on various GPU partitions'
        self.maintainers = ['Craig Meyer']

        #test_name = self.display_name.split(' ')[0]

        sys_info = set_system(config_path, 'gpu_count_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node'] # Maximum number of GPUs available per node
        self.cpus_per_gpu = sys_info['cpus-per-gpu'] # Number of CPUs associated with each GPU

        # The executable is meaningless, but needs to be set for test to run
        # Output from postrun_cmds is what is relevant to the test
        self.executable = 'echo'
        self.executable_opts = ['hello world']
        
        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Job options
        job_info = get_job_options(config_path, 'gpu_count_check')
        self.acct_str = job_info['account']
        if self.access == 'exclusive':
            self.ngpus = self.exclusive_gpus_per_node
            self.exclusive_access = True
        else:
            self.ngpus = 1
        self.num_nodes = job_info['num-nodes']
        self.num_tasks = job_info['num-tasks']
        self.time_limit = job_info['time-limit']
        
        self.tags = {'gpu'}

    # Test parameter(s)
    params = get_test_params(config_path, 'gpu_count_check')
    access = parameter(params['access'])

    # Job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
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

        sys_info = set_system(config_path, 'gpu_cpu_count_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node']
        self.cpus_per_gpu = sys_info['cpus-per-gpu']
        self.all_gpu_nodes = sys_info['all-gpu-nodes'] # String of all nodes in the partition e.g. 'nid[0000, 0001, 0002, ...]'
        self.num_gpu_nodes = len(self.all_gpu_nodes.split(','))

        # The executable is meaningless, but needs to be set for test to run
        # Output from postrun_cmds is what is relevant to the test
        self.executable = 'echo'
        self.executable_opts = ['hello world']
        
        # Set command to list all GPUs available to the job
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Handle job config for both exclusive and shared access
        job_info = get_job_options(config_path, 'gpu_cpu_count_check')
        self.acct_str = job_info['account']
        if self.ngpus == self.exclusive_gpus_per_node:
            self.exclusive_access = True
        self.num_tasks = self.ngpus
        self.nun_nodes = job_info['num-nodes']
        self.time_limit = job_info['time-limit']

        # First one should run successfully, second fail with too many processors
        self.postrun_cmds += [
            f'srun -n {self.num_tasks * self.cpus_per_gpu} -c 1 hostname',
            f'srun -n {self.num_tasks * self.cpus_per_gpu + 1} - c 1 hostname',
            f'scontrol show nodes {self.all_gpu_nodes} | grep Gres= | sort | uniq -c'
        ]
        
        self.tags = {'gpu'}

    # Test parameter(s)
    params = get_test_params(config_path, 'gpu_cpu_count_check')
    ngpus = parameter(params['num-gpus'])

    # Job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
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
class gpu_affinity_check(rfm.RegressionTest):
    def __init__(self):

        self.descr = 'Base test class for GPU affinity tests'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path, 'gpu_affinity_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node'] # Maximum number of GPUs available per node
        self.cpus_per_gpu = sys_info['cpus-per-gpu'] # Number of CPUs associated with each GPU

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
        job_info = get_job_options(config_path, 'gpu_affinity_check')
        self.acct_str = job_info['account']
        if self.access == 'exclusive':
            self.exclusive_access = True
            self.ngpus_per_node = self.exclusive_gpus_per_node
            self.ntasks_per_node = self.exclusive_gpus_per_node
            self.num_tasks = self.exclusive_gpus_per_node
        elif self.access == 'shared':
            self.ngpus_per_node = 2
            self.ntasks_per_node = 2
            self.num_tasks = 2
        self.ngpus_per_task = job_info['num-gpus-per-task']
        self.num_nodes = job_info['num-nodes']
        self.ncpus_per_task = self.cpus_per_gpu

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        
        self.tags = {'gpu'}

    # Test parameter(s)
    params = get_test_params(config_path, 'gpu_affinity_check')
    access = parameter(params['access'])

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
        sys_info = set_system(config_path, 'gpu_affinity_check')
        gpu_dict = sys_info['gpu-cpu-association']

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

        sys_info = set_system(config_path, 'gpu_affinity_array_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node'] # Maximum number of GPUs available per node
        self.cpus_per_gpu = sys_info['cpus-per-gpu'] # Number of CPUs associated with each GPU

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
        job_info = get_job_options(config_path, 'gpu_affinity_array_check')
        self.acct_str = job_info['account']
        self.num_nodes = self.job_config[0]
        self.num_array_tasks = self.job_config[1]
        self.ngpus_per_node = self.job_config[2]
        self.ntasks_per_node = self.job_config[3]
        self.num_tasks = self.ntasks_per_node * self.num_nodes
        self.ngpus_per_task = self.ngpus_per_node // self.ntasks_per_node
        self.ncpus_per_task = self.cpus_per_gpu * self.ngpus_per_task
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        self.tags = {'gpu'}

    # Test parameter(s)
    params = get_test_params(config_path, 'gpu_affinity_array_check')
    job_config = parameter(params['job-array-config'])


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
        sys_info = set_system(config_path, 'gpu_affinity_array_check')
        gpu_dict = sys_info['gpu-cpu-association']

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

        sys_info = set_system(config_path, 'gpu_affinity_jobpacking_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node'] # Maximum number of GPUs available per node
        self.cpus_per_gpu = sys_info['cpus-per-gpu'] # Number of CPUs associated with each GPU
        self.mem_per_gpu = sys_info['mem-per-gpu'] # Memory of each GPU in MB

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
        job_info = get_job_options(config_path, 'gpu_affinity_jobpacking_check')
        self.acct_str = job_info['account']
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
        self.ncpus_per_task = self.cpus_per_gpu * self.ngpus_per_task
        self.mem_1 = self.ngpus_per_node_1 * self.mem_per_gpu
        self.mem_2 = self.ngpus_per_node_2 * self.mem_per_gpu

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
            
        self.tags = {'gpu'}
            
    # Test parameter(s)
    params = get_test_params(config_path, 'gpu_affinity_jobpacking_check')
    job_config = parameter(params['job-packing-config'])
    #job_config = parameter([[params['num-nodes'][i], params['num-gpus-node1'][i], params['num-gpus-node2'][i], params['num-gpus-per-task'][i]] for i in range(8)])

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
        sys_info = set_system(config_path, 'gpu_affinity_jobpacking_check')
        gpu_dict = sys_info['gpu-cpu-association']

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


@rfm.simple_test
class gpu_accounting_check(rfm.RunOnlyRegressionTest):
    def __init__(self):

        # Metadata
        self.descr = 'Test to check the SLURM accounting for GPU jobs'
        self.maintainers = ['Craig Meyer']

        sys_info = set_system(config_path, 'gpu_accounting_check')
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]
        self.exclusive_gpus_per_node = sys_info['exclusive-gpus-per-node'] # Maximum number of GPUs available per node
        self.cpus_per_gpu = sys_info['cpus-per-gpu'] # Number of CPUs associated with each GPU

        # Execution - sleep for 3 minutes
        self.executable = 'sleep'
        self.runtime = 180
        self.executable_opts = [f'{self.runtime}s']

        # Job options
        job_info = get_job_options(config_path, 'gpu_accounting_check')
        self.acct_str = job_info['account']
        self.num_nodes = job_info['num-nodes']
        self.ncpus_per_task = job_info['num-cpus-per-task']
        self.time_limit = job_info['time-limit']
        if self.ngpus == self.exclusive_gpus_per_node:
            self.exclusive_access = True
        self.num_tasks = self.ngpus

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Number of CPUs SLURM will charge given resource request (8 CPUs per GPU, 2 cores per CPU)
        self.ncpus = self.ngpus * self.cpus_per_gpu * 2
        # Pre-decimal Value output from `sacct` command, given runtime and conversion from seconds to hours
        self.val = floor(self.runtime * self.ncpus / 3600 / 2 * 10) / 10

        # Extract job accounting information with sacct
        self.postrun_cmds = ['sacct -X -j $SLURM_JOB_ID --format=CPUTimeRaw | grep -v batch | awk \'{sum=$1}END{print sum/3600/2}\'']
        

    # Test parameter(s)
    params = get_test_params(config_path, 'gpu_accounting_check')
    ngpus = parameter(params['num-gpus'])
    
    # Job options
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--nodes={self.num_nodes}',
            f'--gres=gpu:{self.ngpus}',
            f'--account={self.acct_str}',
        ]

    @run_before('run')
    def set_srun_opts(self):
        self.job.launcher.options = [f'-c {self.ncpus_per_task}']
    
    @sanity_function
    def assert_account_charge(self):
        # Sometimes the actual runtime is > self.runtime by enough to affect self.val calculation (some SLURM overhead?)
        # This is an issue for exclusive, where it only takes 6 extra seconds for the value in the sanity check to change, whereas it 
        # takes over 20 extra seconds when using 2 GPUs. Variance in submission times (anecdotally) seem to be up to 15 seconds
        if self.ngpus == 8:
            return sn.assert_found(str(self.val)[0] + '.', self.stdout)
        else:
            return sn.assert_found(str(self.val), self.stdout)
