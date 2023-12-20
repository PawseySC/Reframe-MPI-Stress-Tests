# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from math import sqrt

# Import functions to set environment variables, etc.
import sys
import os
sys.path.append(os.getcwd() + '/common/scripts')
from set_test_env import *


# Base MPI communications test class
class gpu_mpi_comms_base_check(rfm.RegressionTest):
    def __init__(self, name, **kwargs):

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:work']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to
        self.exclusive_gpus_per_node = 8 # Maximum number of GPUs available per node
        self.cpus_per_gpu = 8 # Number of CPUs associated with each GPU

        # Metadata
        self.descr = 'Base test for GPU-MPI communication tests'
        self.maintainers = ['Craig Meyer']

        # Compilation - build from makefile
        self.build_system = 'Make'
        self.build_system.flags_from_environ = False
        self.executable = 'gpu-mpi-comms'

        # Compile profile_util library used in these tests
        self.prebuild_cmds = [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_hip.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}',
        ]

        # Set job options
        self.num_tasks = self.ngpus_per_node
        if self.ngpus_per_node == self.exclusive_gpus_per_node:
            self.exclusive_access = True
        self.num_cpus_per_task = self.cpus_per_gpu 
        self.time_limit = '10m'

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load), etc.
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = True, omp = iomp, gpu = True)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        self.keep_files = ['logs/*']
        self.tags = {'gpu', 'mpi'}

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    ngpus_per_node = parameter([2, 8])
    num_nodes = parameter([1, 2, 4])

    ###########################
    # RECOMMENDED JOB OPTIONS #
    ###########################
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            '--nodes=1',
            f'--account={self.acct_str}',
            f'--gpus-per-node={self.ngpus_per_node}',
        ]
    # Check node health pre- and post-job
    @run_before('run')
    def check_node_health(self):
        self.prerun_cmds += ['common/scripts/node_check.sh']
        self.postrun_cmds = ['common/scripts/node_check.sh']
    # Explicitly set -c in srun statements (needed for SLURM > 21.08)
    @run_before('run')
    def srun_cpus_per_task(self):
        if self.job.scheduler.registered_name in ['slurm', 'squeue']:
            self.job.launcher.options = [f'-c {self.num_cpus_per_task}']



# GPU -> GPU MPI copy test
@rfm.simple_test
class gpu_copy_check(gpu_mpi_comms_base_check):
    def __init__(self, **kwargs):
        super().__init__('GPUCopy', **kwargs)

        # Metadata
        self.description = 'Test for device MPI GPU -> GPU copy'
        self.maintainers = ['Craig Meyer']

        # Reference value when run with base conditions (one node, one task, etc.)
        self.scaling_factor = 1 + self.ngpus_per_node / 8
        self.reference = {
            '*': {
                'avg time': (6.7e4 * self.scaling_factor, None, 0.2),
                'sd time': (0 if self.ngpus_per_node == 1 else 44000, None, 0.2),
                'min time': (1.5e3 * self.scaling_factor, None, 0.2),
                'max time': (2e6 * self.scaling_factor, None, 0.2),
            },
        }

    # Set dictionary of performance metrics
    @run_before('performance')
    def set_perf_dict(self):

        keys = ['avg time', 'sd time', 'min time', 'max time']
        metrics = ['avg', 'sd', 'min', 'max']
        values = [self.extract_avg_time(kind = i) for i in metrics]

        self.perf_variables = {}
        for key, value in zip(keys, values):
            self.perf_variables[key] = value


    # Test performance (time and bandwidth)
    @performance_function('us')
    def extract_avg_time(self, kind = 'avg'):

        str_stats = sn.extractall(rf'@MPITestGPUCopy.*timing.*max\]=\[(.*)\]\s+\(microseconds\).*', self.stdout, 1)
        str_stats = [time.strip('[] ').split(',') for time in str_stats]
        stats = [float(stat) for stat_lst in str_stats for stat in stat_lst]
        if kind == 'avg':
            return sum(stats[::4]) / len(stats[::4])
        elif kind == 'sd':
            return sqrt((sum([sd**2 for sd in stats[1::4]])) / len(stats[1::4]))
        elif kind == 'min':
            return min(stats[2::4])
        elif kind == 'max':
            return max(stats[3::4])


    # Test passes if the end of the job is reached
    @sanity_function
    def assert_complete(self):
        return sn.assert_found(r'@MPITestGPUCopy.*Reached end of this routine', self.stdout)


# GPU MPI single send/recv performance (bandwidth) test
@rfm.simple_test
class gpu_sendrecv_check(gpu_mpi_comms_base_check):
    def __init__(self, **kwargs):
        super().__init__('GPUSendRecv', **kwargs)

        # Metadata
        self.description = 'Performance (bandwidth) test for GPU MPI single send/recv operations'
        self.maintainers = ['Craig Meyer']

    # Set dictionary of performance metrics
    @run_before('performance')
    def set_perf_dict(self):

        time_keys = ['GPU ' + str(i) + ' avg time' for i in range(self.ngpus_per_node)]
        time_keys += ['GPU ' + str(i) + ' sd time' for i in range(self.ngpus_per_node)]
        time_keys += ['GPU ' + str(i) + ' min time' for i in range(self.ngpus_per_node)]
        time_keys += ['GPU ' + str(i) + ' max time' for i in range(self.ngpus_per_node)]
        bw_keys = ['GPU ' + str(i) + ' avg BW' for i in range(self.ngpus_per_node)]
        bw_keys += ['GPU ' + str(i) + ' sd BW' for i in range(self.ngpus_per_node)]
        bw_keys += ['GPU ' + str(i) + ' min BW' for i in range(self.ngpus_per_node)]
        bw_keys += ['GPU ' + str(i) + ' max BW' for i in range(self.ngpus_per_node)]
        keys = time_keys + bw_keys
        metrics = ['avg', 'sd', 'min', 'max']
        time_values = [self.extract_avg_time(device = i, kind = j) for i in range(self.ngpus_per_node) for j in metrics]
        bw_values = [self.extract_avg_bw(device = i, kind = j) for i in range(self.ngpus_per_node) for j in metrics]
        values = time_values + bw_values

        self.perf_variables = {}
        for key, value in zip(keys, values):
            self.perf_variables[key] = value

    # Test performance (time and bandwidth)
    @performance_function('s')
    def extract_avg_time(self, device = 0, kind = 'avg'):

        str_times = sn.extractall(rf'@MPITestGPUBandwidthSendRecv.*GPU\s+{device}.*:\s+Times\s+\(s\)\s+\[.*\]\s+=\s+(\[.*\]),', self.stdout, 1)
        str_times = [time.strip('[] ').split(',')[:-1] for time in str_times]
        if kind == 'avg':
            times = [float(time_lst[3]) for time_lst in str_times]
            return sum(times) / len(times)
        elif kind == 'sd':
            upp_sigma = [float(time_lst[4]) for time_lst in str_times]
            low_sigma = [float(time_lst[2]) for time_lst in str_times]
            variances = [(u - l)**2 for u in upp_sigma for l in low_sigma]
            return sqrt(sum(variances) / len(variances))
        elif kind == 'min':
            return min([float(time) for time_lst in str_times for time in time_lst])
        elif kind == 'max':
            return max([float(time) for time_lst in str_times for time in time_lst])
    @performance_function('GB/s')
    def extract_avg_bw(self, device = 0, kind = 'avg'):

        str_bws = sn.extractall(rf'@MPITestGPUBandwidthSendRecv.*GPU\s+{device}.*:.*Bandwidth\s+\(GB\/s\)\s+\[.*\]\s+=\s+(\[.*\])', self.stdout, 1)
        str_bws = [bw.strip('[] ').split(',')[:-1] for bw in str_bws]
        if kind == 'avg':
            bws = [float(bw_lst[3]) for bw_lst in str_bws]
            return sum(bws) / len(bws)
        elif kind == 'sd':
            upp_sigma = [float(bw_lst[4]) for bw_lst in str_bws]
            low_sigma = [float(bw_lst[2]) for bw_lst in str_bws]
            variances = [(u - l)**2 for u in upp_sigma for l in low_sigma]
            return sqrt(sum(variances) / len(variances))
        elif kind == 'min':
            return min([float(bw) for bw_lst in str_bws for bw in bw_lst])
        elif kind == 'max':
            return max([float(bw) for bw_lst in str_bws for bw in bw_lst])

    # Test passes if the end of the job is reached
    @sanity_function
    def assert_complete(self):
        return sn.assert_found(r'@MPITestGPUBandwidthSendRecv.*Reached end of this routine', self.stdout)


# GPU MPI single send/recv performance (bandwidth) test
@rfm.simple_test
class gpu_correct_sendrecv_check(gpu_mpi_comms_base_check):
    def __init__(self, **kwargs):
        super().__init__('GPUSendRecv', **kwargs)

        # Metadata
        self.description = 'Test to check correctness of data through GPU MPI single send/recv operations'
        self.maintainers = ['Craig Meyer']


    # Test passes if the end of the job is reached and sends/recvs are correct
    @sanity_function
    def assert_complete(self):
        return sn.all([
            sn.assert_found(r'@MPITestGPUCorrectSendRecv.*Reached end of this routine', self.stdout),
            sn.assert_not_found(r'GOT WRONG data VALUE', self.stdout),
        ])


# GPU MPI all-reduce operation test
@rfm.simple_test
class gpu_allreduce_check(gpu_mpi_comms_base_check):
    def __init__(self, **kwargs):
        super().__init__('GPUAllReduce', **kwargs)

        # Metadata
        self.description = 'Test to check GPU MPI all-reduce operation'
        self.maintainers = ['Craig Meyer']

        # Reference value when run with base conditions (one node, one task, etc.)
        self.scaling_factor = 1 + self.ngpus_per_node / 8
        self.reference = {
            '*': {
                'avg time': (6e4 * self.scaling_factor, None, 0.2),
                'sd time': (2e5 * self.scaling_factor, None, 0.2),
                'min time': (1.2e3 * self.scaling_factor, None, 0.2),
                'max time': (2e5 * self.scaling_factor, None, 0.2),
            },
        }

    
    ngpus_per_node = parameter([2])

    # Set dictionary of performance metrics
    @run_before('performance')
    def set_perf_dict(self):

        keys = ['avg time', 'sd time', 'min time', 'max time']
        metrics = ['avg', 'sd', 'min', 'max']
        values = [self.extract_avg_time(kind = i) for i in metrics]

        self.perf_variables = {}
        for key, value in zip(keys, values):
            self.perf_variables[key] = value


    # Test performance (time and bandwidth)
    @performance_function('us')
    def extract_avg_time(self, kind = 'avg'):

        str_stats = sn.extractall(rf'@MPITestGPUCopy.*timing.*max\]=\[(.*)\]\s+\(microseconds\).*', self.stdout, 1)
        str_stats = [time.strip('[] ').split(',') for time in str_stats]
        stats = [float(stat) for stat_lst in str_stats for stat in stat_lst]
        if kind == 'avg':
            return sum(stats[::4]) / len(stats[::4])
        elif kind == 'sd':
            return sqrt(sum([sd**2 for sd in stats[1::4]]) / len(stats[1::4]))
        elif kind == 'min':
            return min(stats[2::4])
        elif kind == 'max':
            return max(stats[3::4])

    # Test passes if the end of the job is reached
    @sanity_function
    def assert_complete(self):
        return sn.assert_found(r'@MPITestGPUAllReduce.*Reached end of this routine', self.stdout)

# GPU MPI asynchronous send/recv test
@rfm.simple_test
class gpu_async_sendrecv_check(gpu_mpi_comms_base_check):
    def __init__(self, **kwargs):
        super().__init__('GPUAllReduce', **kwargs)

        # Metadata
        self.description = 'Test to check GPU MPI asynchronous send/recv operation'
        self.maintainers = ['Craig Meyer']

        # Reference value when run with base conditions (one node, one task, etc.)
        self.scaling_factor = self.ngpus_per_node
        self.reference = {
            '*': {
                'avg time': (3e5 * self.scaling_factor, None, 0.2),
                'sd time': (2e5 * self.scaling_factor, None, 0.2),
                'min time': (3e2 * self.scaling_factor, None, 0.2),
                'max time': (8e5 * self.scaling_factor, None, 0.2),
            },
        }
    
    ngpus_per_node = parameter([2])

    # Set dictionary of performance metrics
    @run_before('performance')
    def set_perf_dict(self):

        keys = ['avg time', 'sd time', 'min time', 'max time']
        metrics = ['avg', 'sd', 'min', 'max']
        values = [self.extract_avg_time(kind = i) for i in metrics]

        self.perf_variables = {}
        for key, value in zip(keys, values):
            self.perf_variables[key] = value

    # Test performance (time and bandwidth)
    @performance_function('us')
    def extract_avg_time(self, kind = 'avg'):

        str_stats = sn.extractall(rf'@MPITestGPUAsyncSendRecv.*timing.*max\]=\[(.*)\]\s+\(microseconds\).*', self.stdout, 1)
        str_stats = [time.strip('[] ').split(',') for time in str_stats]
        stats = [float(stat) for stat_lst in str_stats for stat in stat_lst]
        if kind == 'avg':
            return sum(stats[::4]) / len(stats[::4])
        elif kind == 'sd':
            return sqrt(sum([sd**2 for sd in stats[1::4]]) / len(stats[1::4]))
        elif kind == 'min':
            return min(stats[2::4])
        elif kind == 'max':
            return max(stats[3::4])

    # Test passes if the end of the job is reached
    @sanity_function
    def assert_complete(self):
        return sn.assert_found(r'@MPITestGPUAsyncSendRecv.*Reached end of this routine', self.stdout)
