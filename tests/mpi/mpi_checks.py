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
from common.scripts.set_test_env import *


# Base MPI communications test class
class MPI_Comms_Base(rfm.RegressionTest):
    def __init__(self, name, **kwargs):

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:work']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to

        # Metadata
        self.descr = 'Performance scaling test for MPI communcation'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        # Compilation
        self.build_system = 'SingleSource'
        # Set appropriate flags
        self.build_system.cppflags = [
            '-fopenmp', '-O3', '-D_MPI',
            '-L${PROFILE_UTIL_DIR}/lib',
            '-I${PROFILE_UTIL_DIR}/include',
            '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/',
            '-lprofile_util_mpi_omp',
        ]
        # Build profile util library used by the source code
        self.prebuild_cmds = [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_cpu.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}'
        ]
        # As tests focus on MPI, here the default is 1 thread per MPI process
        self.num_cpus_per_task = 1
        iomp = True if self.num_cpus_per_task > 1 else False
        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(mpi = True, omp = iomp)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        # Keep log files from node check
        self.keep_files = ['logs/*']


    ###########################
    # RECOMMENDED JOB OPTIONS #
    ###########################
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--account={self.acct_str}',
            f'--nodes{self.num_nodes}'
        ]
    # Explicitly set cpus_per_task in job launcher call - NEEDED FOR SLURM > 21.08 since sbatch option not passed to srun
    @run_before('run')
    def set_cpus_per_task(self):
        if self.job.scheduler.registered_name in ['slurm', 'squeue']:
            self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
    # Check node health pre- and post-job
    @run_before('run')
    def check_node_health(self):
        self.prerun_cmds += ['common/scripts/node_check.sh',]
        self.postrun_cmds = ['common/scripts/node_check.sh',]

    # Test passes if the end of the job is reached
    @sanity_function
    def assert_complete(self):
        return sn.assert_found(r'Job completed at.+', self.stdout)

# Point-to-point communcation test
@rfm.simple_test
class Pt2Pt(MPI_Comms_Base):
    def __init__(self, **kwargs):
        super().__init__('Pt2Pt', **kwargs)

        # Metadata
        self.description = 'Performance/scaling test for MPI point-to-point communication'

        # Compilation - name of source code file
        self.sourcepath = 'pt2pt.cpp'

        # Execution - executable and parameters
        self.executable = 'pt2pt.out'
        # Executable options are `ndata`, `iadjacent`, `iblocking`, `iverbose`, `idelay`, `irandom`
        self.ndata = 512 # Each rank generates `ndata`^3 data
        self.iadjacent = 0 # Ranks communicate all other ranks
        self.iblocking = 3 # Fully asynchronous pt2pt comms
        self.idelay = 0 # Don't add random delays to sends and receives of processes
        self.irandom = 0 # Don't randomise order of processes
        self.executable_opts = [f'{self.ndata} {self.iadjacent} {self.iblocking} 1 {self.idelay} {self.irandom}']

        # Job options
        self.num_tasks_per_node = 24
        self.num_tasks = self.num_nodes * self.num_tasks_per_node

        # Reference value when run with base conditions (one node, one task, etc.)
        self.ref_val = 2e6
        # Dictionary holding the reference values to use in the performance test
        scaling_factor = 1 / self.num_nodes if self.num_nodes > self.num_tasks_per_node else 1 / self.num_tasks_per_node
        self.reference = {
            'system:work': {'Average': (self.ref_val * scaling_factor, None, 0.2)},
        }

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    num_nodes = parameter([1, 2, 4, 8, 16, 32])

    # Performance function for the recorded time statistics
    @performance_function('us')
    def extract_timing(self, kind='Average'):
        if kind not in ('Average', 'Standard Deviation', 'Maximum', 'Minimum', 'IQR'):
            raise ValueError(f'Illegal value in argument kind ({kind!r})')
        
        # Extract timing for the redistribution of data, as that is where the 
        # point-to-point communcation is. The generation of data does not
        # include any point-to-point communication amongst ranks
        return sn.extractsingle(rf'@redistributeData.+{kind}\s=\s(\S+),.+', 
                                self.stdout, 1, float)
    

# Collective communication test
@rfm.simple_test
class CollectiveComms(MPI_Comms_Base):
    def __init__(self, **kwargs):
        super().__init__('CollectiveComms', **kwargs)

        # Metadata
        self.descr = 'Performance/scaling test for MPI collective communication'

        # Compilation - source code file
        self.sourcepath = 'collective.cpp'

        # Execution - executable and parameters
        # Parameters control number of data points and verbosity of output
        self.executable = 'collective.out'
        self.ndata = 100
        self.executable_opts = [f'{self.ndata} 1']

        # Job options
        self.num_tasks_per_node = 24
        self.num_tasks = self.num_nodes * self.num_tasks_per_node

        # Reference value when run with base conditions (one node, one task, etc.)
        self.ref_vals = {
            'bcast': 3e6,
            'gather': 2e6,
            'distribute': 1.5e6,
            'reduce': 6e3,
        }
        # Dictionary holding the reference values to use in the performance test
        scaling_factor = 1 / self.num_nodes if self.num_nodes > self.num_tasks_per_node else 1 / self.num_tasks_per_node
        self.reference = {
            'system:work': {
                'bcast_Average': (self.ref_vals['bcast'] * scaling_factor, None, 0.2),
                'gather_Average': (self.ref_vals['gather'] * scaling_factor, None, 0.2),
                'distribute_Average': (self.ref_vals['distribute'] * scaling_factor, None, 0.2),
                'reduce_Average': (self.ref_vals['reduce'] * scaling_factor, None, 0.2),
            },
        }
    
    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    num_nodes = parameter([1, 2, 4, 8, 16, 32])

    # Return the average, SD, min and max times, and inter-quartile range
    # using the performance function (defined uniquely in each subtest)
    @run_before('performance')
    def set_perf_dict(self):
        self.perf_variables = {
            'bcast_Average': self.extract_timing('broadcastData'),
            'bcast_Standard Deviation': self.extract_timing('broadcastData', 'Standard Deviation'),
            'bcast_Mininum': self.extract_timing('broadcastData', 'Minimum'),
            'bcast_Maximum': self.extract_timing('broadcastData', 'Maximum'),
            'bcast_IQR': self.extract_timing('broadcastData', 'IQR'),
            'gather_Average': self.extract_timing('gatherData'),
            'gather_Standard Deviation': self.extract_timing('gatherData', 'Standard Deviation'),
            'gather_Minimum': self.extract_timing('gatherData', 'Minimum'),
            'gather_Maximum': self.extract_timing('gatherData', 'Maximum'),
            'gather_IQR': self.extract_timing('gatherData', 'IQR'),
            'distribute_Average': self.extract_timing(),
            'distribute_Standard Deviation': self.extract_timing(kind = 'Standard Deviation'),
            'distribute_Minimum': self.extract_timing(kind = 'Minimum'),
            'distribute_Maximum': self.extract_timing(kind = 'Maximum'),
            'distribute_IQR': self.extract_timing(kind = 'IQR'),
            'reduce_Average': self.extract_timing('reduceData', 'Average'),
            'reduce_Standard_Deviation': self.extract_timing('reduceData', 'Standard Deviation'),
            'reduce_Minimum': self.extract_timing('reduceData', 'Minimum'),
            'reduce_Maximum': self.extract_timing('reduceData', 'Maximum'),
            'reduce_IQR': self.extract_timing('reduceData', 'IQR'),
        }

    # Performance function for the recorded time statistics
    @performance_function('us')
    def extract_timing(self, func = 'redistributeData2', kind='Average'):
        if kind not in ('Average', 'Standard Deviation', 'Minimum', 'Maximum', 'IQR'):
            raise ValueError(f'Illegal value in argument kind ({kind!r})')
        
        # Extract timing for the passed function
        return sn.extractsingle(rf'@{func}.+{kind}\s=\s(\S+),.+', self.stdout, 1, float)

# Test for hang due to large delay in send and receive
@rfm.simple_test
class DelayHang(MPI_Comms_Base):
    def __init__(self, **kwargs):
        super().__init__('DelayHang', **kwargs)

        # Metadata
        self.descr = 'Test to check MPI hangs observed in ASKAP workflow'

        # Compilation - source code file
        self.sourcepath = 'misc_tests.cpp'

        # Execution - executable and arguments
        self.executable = 'misc_tests.out'
        # Executable options set number of data, mode of operation, which rank has delay, how long delay is
        self.ndata = 512
        self.delay_rank = 45
        self.delay_time = 600 # seconds
        self.executable_opts = [f'{self.ndata} HANGING {self.delay_rank} {self.delay_time}']

        # Job options
        self.num_nodes = 2
        self.num_tasks_per_node = 128
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.time_limit = self.delay_time + 60 # Set job time limit to delay + 1 minute
    

# Test employing ucx library
@rfm.simple_test
class CorrectSends(rfm.RegressionTest):
    def __init__(self):
        
        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:work']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to

        # Metadata
        self.descr = 'Test to check MPI sends are correct'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        # Compilation - source code file
        self.sourcepath = 'misc_tests.cpp'
        self.build_system = 'SingleSource'
        self.build_system.cppflags = ['-fopenmp', '-O3', '-D_MPI']

        # Execution - executable and arguments
        self.executable = 'misc_tests.out'
        # Executable options control number of data points, mode of operation, and type of MPI send
        self.ndata = 100
        self.executable_opts = [f'{self.ndata} CORRECT_SENDS {self.send_mode}']
        self.keep_files = ['logs/*']

        # Run the test using the `craype-network-ucx` module
        # self.modules = ['craype-network-ucx', 'cray-mpich-ucx']
        # Needed for ucx module(s) to work properly
        # self.prebuild_cmds = ['module unload xpmem']

        # Job options
        self.num_nodes = 1
        self.num_tasks = 24
        self.num_cpus_per_task = 1

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = True, omp = iomp)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    send_mode = parameter(['isend', 'send', 'ssend'])


    ###########################
    # RECOMMENDED JOB OPTIONS #
    ###########################
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--account={self.acct_str}',
            f'--nodes={self.num_nodes}'
        ]
    # Run many iterations
    @run_before('run')
    def iterate_run(self):
        cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds += [
            f'for ((i=0;i<50;i++)); ' + 
            f'do {cmd} -N {self.num_nodes} -n {self.num_tasks} -c 1 {self.executable} {self.executable_opts[0]}; ' +
            'done'
        ]
    # Explicitly set cpus_per_task in job launcher call - NEEDED FOR SLURM > 21.08 since sbatch option not passed to srun
    @run_before('run')
    def set_cpus_per_task(self):
        if self.job.scheduler.registered_name in ['slurm', 'squeue']:
            self.job.launcher.options = [f'-c {self.num_cpus_per_task}']
    # Check node health pre- and post-job
    @run_before('run')
    def check_node_health(self):
        self.prerun_cmds += ['common/scripts/node_check.sh',]
        self.postrun_cmds = ['common/scripts/node_check.sh',]
    
    # Test passes if the end of the job is reached
    @sanity_function
    def assert_complete(self):
        return sn.all([
            sn.assert_found(r'Job completed at.+', self.stdout),
            sn.assert_not_found(r'.*WRONG.*', self.stdout)
        ])
        

# Test for hang and crashes due to large comm world + async communication
@rfm.simple_test
class LargeCommHang(MPI_Comms_Base):
    def __init__(self, **kwargs):
        super().__init__('LargeCommHang', **kwargs)

        # Metadata
        self.descr = 'Test for large comm-world hang with MPI codes'

        # Compilation
        self.sourcepath = 'pt2pt.cpp'

        # Execution
        self.executable = 'pt2pt.out'
        self.ndata = 1024 # Each rank generates `ndata^3` data
        self.iadjacent = 0 # Ranks communicate with all other ranks
        self.iblocking = 3 # Fully asynchronous MPI pt2pt communication
        self.idelay = 0 # No delays between sends and receives for ranks
        self.irandom = 0 # Don't randomise order of ranks
        self.executable_opts = [f'{self.ndata} {self.iadjacent} {self.iblocking} 1 {self.idelay} {self.irandom}']

        # Job options
        self.num_nodes = 4
        self.num_tasks_per_node = self.ntasks_per_node
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.exclusive_access = True
        self.time_limit = 300 # Set job time limit to 5 minutes

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    ntasks_per_node = parameter([85, 86])


# Test memory usage of MPI code and compare to system consumed memory
@rfm.simple_test
class MemoryLeak(rfm.RegressionTest):
    def __init__(self):

        #############################################################
        # THESE OPTIONS ARE SITE/SYSTEM-SPECIFIC AND NEED TO BE SET #
        #############################################################
        self.valid_systems = ['system:work']
        self.valid_prog_environs = ['*']
        self.acct_str = 'account_name' # Account to charge job to

        # Metadata
        self.descr = 'Test memory sampling/reporting during MPI comms'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        # Compilation
        self.sourcepath = 'mpi-comms.cpp'
        self.build_system = 'SingleSource'
        self.build_system.cppflags = [
            '-fopenmp', '-O3', '-D_MPI',
            '-L${PROFILE_UTIL_DIR}/lib',
            '-I${PROFILE_UTIL_DIR}/include',
            '-Wl,-rpath=${PROFILE_UTIL_DIR}/lib/',
            '-lprofile_util_mpi_omp',
        ]
        # Build profile util library used by the source code
        self.prebuild_cmds = [
            'MAIN_SRC_DIR=$(pwd)',
            'cd common/profile_util',
            './build_cpu.sh',
            'PROFILE_UTIL_DIR=$(pwd)',
            'cd ${MAIN_SRC_DIR}'
        ]
        # Compile process tracking program
        flags_str = ' '.join(self.build_system.cppflags)
        self.prebuild_cmds += ['CC ' + flags_str + ' get_running_procs.cpp -o get_running_procs.out']

        # Executable
        self.executable = 'mpi-comms.out'
        self.ndata = 300 # Controls number of data generated
        self.executable_opts = [f'{self.ndata} 1 >> mem_reports.log &']
        self.keep_files = ['logs/*']

        # Job options
        self.num_tasks_per_node = self.ntasks_per_node
        self.num_tasks = self.num_tasks_per_node * self.num_nodes
        self.num_cpus_per_task = 1
        self.mem_per_cpu = 1840 # Set to DefMemPerCPU
        self.mem_per_node = self.num_tasks_per_node * self.mem_per_cpu
        self.exclusive_access = True

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        iomp = True if self.num_cpus_per_task > 1 else False
        env_vars, modules, cmds = set_env(mpi = True, omp = iomp)
        self.variables = {env.split('=')[0]: env.split('=')[1] for env in env_vars}
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

        # Cadence of memory reporting (in seconds)
        self.cadence = 1.0 
        # Run memory tracking program as a simultaneous job step alongside main MPI comms program
        self.postrun_cmds = [
            f'srun --exact -N {self.num_nodes} -n {self.num_nodes} --ntasks-per-node=1 -c {self.num_cpus_per_task} --mem={self.mem_per_cpu} '
            + f'./get_running_procs.out {self.cadence} {self.num_tasks - self.num_nodes} >> mem_reports.log &', 
            'wait',
            f'python3 parse_memory.py -n {self.num_tasks} -N {self.num_nodes} -f mem_reports.log',
        ]

    ###########################################
    # SET PARAMETER(S) TO YOUR DESIRED VALUES #
    ###########################################
    num_nodes = parameter([1, 2, 4, 8, 16, 32])
    ntasks_per_node = parameter([24])


    # Modify job launcher options for multiple simultaneous job steps
    @run_before('run')
    def modify_launcher(self):
        self.job.launcher.options = [
            '--exact', 
            f'-N {self.num_nodes}', f'-n {self.num_tasks - self.num_nodes}', f'--ntasks-per-node={self.ntasks_per_node - 1}',
            f'-c {self.num_cpus_per_task}', f'--mem={self.mem_per_node - self.mem_per_cpu}'
        ]
    ###########################
    # RECOMMENDED JOB OPTIONS #
    ###########################
    # NOTE: These don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--mem={self.mem_per_node}',
            f'--account={self.acct_str}',
            f'--nodes={self.num_nodes}',
        ]
    @run_before('run')
    def check_node_health(self):
        self.prerun_cmds += ['common/scripts/node_check.sh',]
        self.postrun_cmds += ['common/scripts/node_check.sh',]
    
    # Test passes if the end of the job is reached
    @sanity_function
    def assert_complete(self):
        # Get nodes running processes
        nodes = sn.evaluate(sn.findall(r'Running on node\s(\w+)', 'mem_reports.log'))
        passfail_list = [False for _ in range(self.num_nodes)]
        # print(nodes)
        # Iterate through the nodes
        for inode in range(self.num_nodes):
            node_id = nodes[inode].groups()[0]
            # Get base system memory consumption
            # Total, Used, Free, Shared, Cache, Available [bytes]
            base_mem = sn.evaluate(
                sn.findall(f'System memory on node {node_id}: ' +
                           r'Mem:\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)', 
                           'mem_reports.log'))[0]
            base_used = float(base_mem.groups()[1])
            # Parse output from python analysis script
            mem_comparisons = sn.evaluate(sn.findall(
                f'ON NODE {node_id} ' + r'>>\sIN FUNCTION\s(\w+) >> Memory used by processes:\s+([0-9]+.[0-9]+)\s+>>\s' +
                r'System memory used:\s+([0-9]+.[0-9]+)', self.stdout
            ))
            ncomparisons = len(mem_comparisons)

            # Lists for use in testing whether this node passed test condition
            func = ["" for _ in range(ncomparisons)]
            proc_mem = [0 for _ in range(ncomparisons)]
            used_mem = [0 for _ in range(ncomparisons)]
            for i in range(ncomparisons):
                func[i] = mem_comparisons[i].groups()[0]
                proc_mem[i] = float(mem_comparisons[i].groups()[1]) + (base_used / (1024 * 1024))
                used_mem[i] = float(mem_comparisons[i].groups()[2])
            # Need list of bools for `all` function
            # Condition is ([used] > [code + OS] by no more than 10%)
            bool_list = [used_mem[i] / proc_mem[i] < 1.1 for i in range(ncomparisons)]
            # Test passes on this node if all elements of `bool_list` are True
            passfail_list[inode] = all(bool_list)

            # Extra output for ReFrame to display at runtime
            with open('passfail.log', 'a') as f:
                for i in range(ncomparisons):
                    f.write('Node %s, Function %s, Code mem + OS = %.3f, `Used` quoted by `free` = %.3f, ratio passed = %s\n' %
                            (node_id, func[i], proc_mem[i], used_mem[i], bool_list[i]))
                f.write('**************************************************\n')

            # Summary statement to show in ReFrame output
            if all(bool_list):
                print("Memory consistent on node ", node_id)
            else:
                print("Memory inconsistent on node ", node_id)

        # Test passes if program runs and there are no disparities between
        # memory used by processes and listed as `used` in the system
        return sn.all([
            sn.assert_found(r'Memory sampling completed at.+', 'mem_reports.log'),
            sn.all(passfail_list)
        ])
