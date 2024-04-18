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
config_path = curr_dir + '/mpi_config.yaml'

# Base MPI communications test class
class MPI_Comms_Base(rfm.RegressionTest):
    def __init__(self, name, **kwargs):

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

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

        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds
        # Keep log files from node check
        self.keep_files = ['logs/*']

        # Set job options
        job_info = get_job_options(config_path)
        self.acct_str = job_info['account']
        self.num_cpus_per_task = 1


    # Set job options for job script
    # NOTE: These job options don't have automatic ReFrame equivalent, so need to be manually set
    # NOTE: Default format is SLURM/SBATCH, adjust if needed
    @run_before('run')
    def set_job_opts(self):
        self.job.options = [
            f'--account={self.acct_str}',
            f'--nodes={self.num_nodes}'
        ]
    # Explicitly set number of CPUs per task in job launcher - NEEDED FOR SLURM
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

        # Get test-specific configuration
        test_config = configure_test(config_path, 'Pt2Pt')

        # Compilation - name of source code file
        self.sourcepath = 'pt2pt.cpp'

        # Execution - executable and parameters
        self.executable = 'pt2pt.out'
        self.executable_opts = [' '.join([v for v in test_config['executable-options'].values()])]

        # Set sbatch script directives
        self.num_tasks_per_node = test_config['job-options']['num-tasks-per-node']
        self.num_tasks = self.num_nodes * self.num_tasks_per_node

        # Reference value when run with base conditions (one node, one task, etc.)
        self.ref_val = test_config['performance']['reference-value']
        # Dictionary holding the reference values to use in the performance test
        # Adjust calculation of scaling depending on varying parameter
        # NOTE: Scaling factor still needs some tuning
        scaling_factor = 1 / self.num_nodes if self.num_nodes > self.num_tasks_per_node else 1 / self.num_tasks_per_node
        self.reference = {
            'system:work': {'Average': (self.ref_val * scaling_factor, None, 0.2)},
        }

    # Test parameter(s)
    params = get_test_params(config_path, 'Pt2Pt')
    num_nodes = parameter(params['num-nodes'])

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

        # Get test-specific configuration
        test_config = configure_test(config_path, 'CollectiveComms')

        # Compilation - source code file
        self.sourcepath = 'collective.cpp'

        # Execution - executable and parameters
        # Parameters control number of data points and verbosity of output
        self.executable = 'collective.out'
        self.executable_opts = [' '.join([v for v in test_config['executable-options'].values()])]

        # Set sbatch script directives
        self.num_tasks_per_node = test_config['job-options']['num-tasks-per-node']
        self.num_tasks = self.num_nodes * self.num_tasks_per_node

        # Reference value when run with base conditions (one node, one task, etc.)
        self.ref_vals = test_config['performance']['reference-value']
        # Dictionary holding the reference values to use in the performance test
        # Adjust calculation of scaling depending on varying parameter
        scaling_factor = 1 / self.num_nodes if self.num_nodes > self.num_tasks_per_node else 1 / self.num_tasks_per_node
        # NOTE: These still need to be adjusted, as does above scaling factor
        self.reference = {
            'system:work': {
                'bcast_Average': (self.ref_vals['bcast'] * scaling_factor, None, 0.2),
                'gather_Average': (self.ref_vals['gather'] * scaling_factor, None, 0.2),
                'distribute_Average': (self.ref_vals['distribute'] * scaling_factor, None, 0.2),
                'reduce_Average': (self.ref_vals['reduce'] * scaling_factor, None, 0.2),
            },
        }
    
    # Test parameter(s)
    params = get_test_params(config_path, 'CollectiveComms')
    num_nodes = parameter(params['num-nodes'])

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

        # Get test-specific configuration
        test_config = configure_test(curr_dir + '/mpi_config.yaml', 'DelayHang')

        # Compilation - source code file
        self.sourcepath = 'misc_tests.cpp'

        # Execution - executable and arguments
        self.executable = 'misc_tests.out'
        # Executable options set number of data, mode of operation, which rank has delay, how long delay is
        self.executable_opts = [' '.join([v for v in test_config['executable-options'].values()])]

        # sbatch script directives
        self.num_nodes = test_config['job-options']['num-nodes']
        self.num_tasks_per_node = test_config['job-options']['num-tasks-per-node']
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.time_limit = int(test_config['executable-options']['delay-time']) + 60 # Set job time limit to delay + 1 minute
    

# Test employing ucx library
@rfm.simple_test
class CorrectSends(rfm.RegressionTest):
    def __init__(self):
        
        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Metadata
        self.descr = 'Test to check MPI sends are correct'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        # Get test-specific configuration
        test_config = configure_test(curr_dir + '/mpi_config.yaml', 'CorrectSends')

        # Compilation - source code file
        self.sourcepath = 'misc_tests.cpp'
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

        # Execution - executable and arguments
        self.executable = 'misc_tests.out'
        # Executable options control number of data points, mode of operation, and type of MPI send
        self.executable_opts = [' '.join([v for v in test_config['executable-options'].values()]) + f' {self.send_mode}']
        self.keep_files = ['logs/*']

        # Run the test using the `craype-network-ucx` module
        self.modules = ['craype-network-ucx', 'cray-mpich-ucx']

        # sbatch script directives
        job_info = get_job_options(config_path)
        self.acct_str = job_info['account']
        self.num_nodes = test_config['job-options']['num-nodes']
        self.num_tasks = test_config['job-options']['num-tasks']
        self.num_cpus_per_task = 1

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(config_path)
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds

    # Test parameter(s)
    params = get_test_params(curr_dir + '/mpi_config.yaml', 'CorrectSends')
    send_mode = parameter(params['send-mode'])


    # Set job options
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

        # Get test-specific configuration
        test_config = configure_test(curr_dir + '/mpi_config.yaml', 'LargeCommHang')

        # Compilation
        self.sourcepath = 'pt2pt.cpp'

        # Execution
        self.executable = 'pt2pt.out'
        self.executable_opts = [' '.join([v for v in test_config['executable-options'].values()])]

        # sbatch script directives
        self.num_nodes = test_config['job-options']['num-nodes']
        #self.num_tasks_per_node = self.ntasks_per_node
        self.num_tasks = self.num_nodes * self.ntasks_per_node
        self.exclusive_access = test_config['job-options']['exclusive']
        self.time_limit = test_config['job-options']['time-limit']

    # Test parameter(s)
    params = get_test_params(curr_dir + '/mpi_config.yaml', 'LargeCommHang')
    ntasks_per_node = parameter(params['num-tasks-per-node'])


# Test memory usage of MPI code and compare to system consumed memory
@rfm.simple_test
class MemoryLeak(rfm.RegressionTest):
    def __init__(self):

        sys_info = set_system(config_path)
        # Valid systems and PEs test will run on
        self.valid_systems = [s for s in sys_info['system']]
        self.valid_prog_environs = [pe for pe in sys_info['prog-environ']]

        # Metadata
        self.descr = 'Test memory sampling/reporting during MPI comms'
        self.maintainers = ['Craig Meyer', 'Pascal Jahan Elahi']

        # Get test-specific configuration
        test_config = configure_test(curr_dir + '/mpi_config.yaml', 'MemoryLeak')

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
        self.executable_opts = [' '.join([v for v in test_config['executable-options'].values()]) + ' >> mem_reports.log &']
        self.keep_files = ['logs/*']

        # Job batch resource allocation specifications
        job_info = get_job_options(config_path)
        self.acct_str = job_info['account']
        self.num_tasks_per_node = self.ntasks_per_node
        self.num_tasks = self.num_tasks_per_node * self.num_nodes
        self.mem_per_cpu = test_config['job-options']['mem-per-cpu']
        self.mem_per_node = self.num_tasks_per_node * self.mem_per_cpu
        self.exclusive_access = test_config['job-options']['exclusive']
        self.num_cpus_per_task = 1

        # Set up environment (any environment variables to set, prerun_cmds, and/or modules to load)
        env_vars, modules, cmds = set_env(curr_dir + '/mpi_config.yaml')
        self.variables = env_vars
        if modules != []:
            self.modules = modules
        if cmds != []:
            self.prerun_cmds = cmds


        # Cadence of memory reporting (in seconds)
        self.cadence = test_config['cadence']
        # Run memory tracking program as a simultaneous job step alongside main MPI comms program
        self.postrun_cmds = [
            f'srun --exact -N {self.num_nodes} -n {self.num_nodes} --ntasks-per-node=1 -c {self.num_cpus_per_task} --mem={self.mem_per_cpu} '
            + f'./get_running_procs.out {self.cadence} {self.num_tasks - self.num_nodes} >> mem_reports.log &', 
            'wait',
            f'python3 parse_memory.py -n {self.num_tasks} -N {self.num_nodes} -f mem_reports.log',
        ]

    # Test parameter(s)
    params = get_test_params(curr_dir + '/mpi_config.yaml', 'MemoryLeak')
    num_nodes = parameter(params['num-nodes'])
    ntasks_per_node = parameter([params['num-tasks-per-node']])


    # Modify job launcher options for multiple simultaneous job steps
    @run_before('run')
    def modify_launcher(self):
        self.job.launcher.options = [
            '--exact', 
            f'-N {self.num_nodes}', f'-n {self.num_tasks - self.num_nodes}', f'--ntasks-per-node={self.ntasks_per_node - 1}',
            f'-c {self.num_cpus_per_task}', f'--mem={self.mem_per_node - self.mem_per_cpu}'
        ]
    # Set job options
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
