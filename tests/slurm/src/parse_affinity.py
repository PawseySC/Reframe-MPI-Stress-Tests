import argparse
import re
from math import ceil, floor

# Custom colour printing for easier interpretation of ReFrame
# `self.stdout` which the output of this script goes
class colours:
    GREEN = '\033[92m' # success
    RED = '\033[91m' # fail
    END = '\033[00m' # end colour region

# Function to print coloured messages
# Usually for marking success (green) and failure (red)
# `colour` parameter must be of form `colours.COLOUR`
def print_colour(message, colour):
    print('{}{}{}'.format(colour, message, colours.END))

# Convert string representation of bool to boolean
# Mainly used for parsing bools passed as command line
# arguments, which are initially read in as strings
def string_to_bool(s):
    return s.upper() == 'TRUE'

# Convert job step statistics from strings to appropriate datatyps
# and do sanity checking after initial parsing through `argparse`
def parse_params_list(lst):

    # Actual list is contained within a list after
    # being processed through `argparse`
    lst = lst[0]
    # Convert elements from string to appropriate data types
    for idx, item in enumerate(lst):
        if idx in [0,2,5]:
            lst[idx] = int(item)
        elif idx == 1:
            lst[idx] = string_to_bool(item)

    # Check values for `OMP_PROC_BIND` and `OMP_PLACES` are valid
    assert (lst[3].upper() in ['MASTER', 'CLOSE', 'SPREAD', 'AUTO', 'TRUE', 'FALSE']), 'ERROR: Invalid value for `OMP_PROC_BIND` parameter'
    assert (lst[4].upper() in ['THREADS', 'CORES', 'SOCKETS']), 'ERROR: Invalid valud for `OMP_PLACES` parameter'

    return lst


# Parse command line arguments using `argparse` package
# NOTE: Node argument may not be necessary - may remove later
def parse_args():

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--nodes', type = int, help = 'Number of nodes')
    parser.add_argument('-p', '--params', nargs = '+',
        help = 'Comma delimited list of job parameters The order of the fields are ' +
               '[num_tasks_per_node, hyperthreading, num_threads, OMP_PROC_BIND, OMP_PLACES, cpus_per_task]',
        type = lambda s: [item for item in s.split(',')], required = True)
    parser.add_argument('-s', '--sys_list', nargs = '+',
        help = 'Comma delimited list of system configuration. The order of the fields are ' +
               '[num_sockets, num_cpus, cpus_per_core, cpus_per_socket]',
        type = lambda s: [int(item) for item in s.split(',')], required = True)
    parser.add_argument('-f', '--file', help = 'Affinity file for job', required = True)
    parser.add_argument('-m', '--mode', help = 'Mode for parsing script (checking either OMP or MPI affinity',
        choices = ['OMP', 'MPI'])
    args = parser.parse_args()

    # Process list arguments
    args.params = parse_params_list(args.params)
    args.sys_list = args.sys_list[0]

    # Return arguments as dictionaries
    job_dict = {
        'num_nodes': args.nodes,
        'num_tasks_per_node': args.params[0],
        'hyperthread': args.params[1],
        'omp_num_threads': args.params[2],
        'omp_proc_bind': args.params[3].upper(),
        'omp_places': args.params[4].upper(),
        'num_cpus_per_task': args.params[5],
        'affinity_file': args.file,
    }
    sys_dict = {
        'num_sockets': args.sys_list[0],
        'num_cpus': args.sys_list[1],
        'num_cpus_per_core': args.sys_list[2],
        'num_cpus_per_socket': args.sys_list[3],
    }

    return sys_dict, job_dict, args.mode


def printConfig(sys_config, job_config):

    print('--------------------------------------------------')
    print('The value of OMP_PROC_BIND is', job_config['omp_proc_bind'])
    print('The value of OMP_PLACES is', job_config['omp_places'])
    print('The number of OMP threads is', job_config['omp_num_threads'])
    print('Hyperthreading =', job_config['hyperthread'])
    print('This job is using', job_config['num_nodes'], 'nodes, with', job_config['num_tasks_per_node'], 
          'tasks per node, each of those tasks with', job_config['num_cpus_per_task'], 'CPUs per task')
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print('When hyperthreading this system has', sys_config['num_sockets'], 'sockets, with',
          sys_config['num_cpus_per_socket'], 'CPUs per socket for a total of', sys_config['num_cpus'], 'CPUs')
    print('When NOT hyperthreading this system has', sys_config['num_sockets'], 'sockets, with',
          sys_config['num_cpus_per_socket'] // sys_config['num_cpus_per_core'], 'CPUs per socket, for a total of',
          sys_config['num_cpus'] // sys_config['num_cpus_per_core'], 'CPUs')
    print('--------------------------------------------------')

# Extract affinity information from file for logic testing
def extract_affinity(job_config):

    # Regex pattern to match affinity output
    # Extract the MPI rank, the OMP thread ID, and the core the thread is placed on
    patt = r'Node\s+(nid[0-9]+).+MPI\s+([0-9]+).+Thread\s+([0-9]+).+\s+placement\s+=\s+([0-9]+)'
    #patt = r'MPI RANK\s+([0-9]+).+OMP\s+([0-9]+).+HWT\s+([0-9]+)'
    #patt = r'MPI RANK\s+([0-9]+).+HOST\s+(\w+[0-9]+).+OMP\s+([0-9]+).+HWT\s+([0-9]+)'


    # Pertinent info from the job configuration dictionary
    num_nodes = job_config['num_nodes']
    ntasks = job_config['num_tasks_per_node']
    nthreads = job_config['omp_num_threads']
    afile = job_config['affinity_file']

    # Lists for holding information for each thread
    nodes = [None for _ in range(nthreads * ntasks * num_nodes)]
    ranks = [None for _ in range(nthreads * ntasks * num_nodes)]
    tids = [None for _ in range(nthreads * ntasks * num_nodes)]
    cores = [None for _ in range(nthreads * ntasks * num_nodes)]

    # Open affinity file for reading
    f = open(afile, 'r')
    lines = f.readlines()
    idx = 0
    # Iterate over each line, skipping those without affinity reporting information
    for line in lines:
        match = re.match(patt, line)
        if match == None: # This line contains no affinity reporting information
            continue
        # From the regex match groups, extract the quantities of interest
        ranks[idx] = int(match.groups()[1])
        nodes[idx] = match.groups()[0]
        tids[idx] = int(match.groups()[2])
        cores[idx] = int(match.groups()[3])
        idx += 1
    # Close the file
    f.close()

    # Check that TIDs are uniquely labelled and that there is the right amount of unique labels
    # If either of these isn't satisifed then that indicates something significantly wrong
    assert (len(set(tids)) * ntasks * num_nodes == len(tids)), 'ERROR: Each OMP thread is not uniquely labelled'
    assert (len(set(tids)) == job_config['omp_num_threads']), 'ERROR: Number of OMP threads does not match $OMP_NUM_THREADS env var'

    # Return all the affinity information (TIDs and associated core for each OMP thread of each MPI rank)
    return nodes, ranks, tids, cores


# Get the socket IDs for each core in `cores`
def getSocketID(hyperthreading, cores, ncpus_per_socket, ncpus_per_core):

    if hyperthreading:
        return [cid // (ncpus_per_socket // ncpus_per_core) % ncpus_per_core + 1 for cid in cores]
    else:
        return [cid // ncpus_per_socket + 1 for cid in cores]


# Check that the reported affinity is sensible given the
# system and OMP environment configuration/setup
def check_affinity(sys_config, job_config, tids, cores):

    # Retrieve relevant fields from dictionaries
    nthreads = job_config['omp_num_threads']
    proc_bind = job_config['omp_proc_bind']
    places = job_config['omp_places']
    hyperthreading = job_config['hyperthread']
    ncpus_per_task = job_config['num_cpus_per_task']
    # Adjust number of CPUs per socket accordingly depending on `--threads-per-core`
    if hyperthreading:
        ncpus_per_socket = sys_config['num_cpus_per_socket']
        ncpus_per_core = sys_config['num_cpus_per_core']
    else:
        ncpus_per_socket = sys_config['num_cpus_per_socket'] // sys_config['num_cpus_per_core']
        ncpus_per_core = 1

    # Get assigned core of master thread for use in logic checking
    # Threads are not necessarily in numerical order in `tids`
    master_idx = tids.index(0)
    master_cid = cores[master_idx]


    # Logic handling for when OMP_PROC_BIND=MASTER
    # Ensure all OMP threads are on the same thread/core/socket as the master thread
    if proc_bind == 'MASTER':
        if places == 'THREADS':
            # For MASTER+THREADS, all OMP threads should be on the same thread, whether hyperthreading or not
            if len(set(cores)) > 1:
                print("PROC: Master, PLACES: Threads - All threads should be on the same thread")
                return False
            return True
        elif places == 'CORES':
            # For MASTER+CORES, if hyperthreading, OMP threads can be on either thread of the core
            # master thread is on. The two threads on the master core have IDs separated by `ncpus_per_socket`
            # If not hyperthreading, all OMP threads will be on the same thread
            if (len(set(cores)) > ncpus_per_core) or (max(set(cores)) - min(set(cores)) > ncpus_per_socket):
                print('PROC: Master, PLACES: Cores - All threads should be on the same core')
                return False
            return True
        elif places == 'SOCKETS':
            # FOR MASTER+SOCKETS all OMP threads need to be on same socket as the master thread
            # Socket of master thread needs to account for SMT when hyperthreading
            if len(set(getSocketID(hyperthreading, cores, ncpus_per_socket, ncpus_per_core))) != 1:
                print('PROC:, Master, PLACES: Sockets - All threads should be on the same socket')
                return False
            return True
    # Logic handling for when OMP_PROC_BIND=CLOSE
    # Ensure OMP threads assignments are contiguous/sequential and don't go above upper bounds
    # based on balance between `nthreads` (OMP_NUM_THREADS) and `ncpus_per_task`
    elif proc_bind == 'CLOSE':
        # For CLOSE+THREADS, OMP threads should be placed on adjacent hardware threads. If there is hyperthreading
        # both threads on a core should be filled before moving onto the next core
        if places == 'THREADS':
            # If nthreads <= ncpus_per_task, each OMP thread will be on a unique thread, otherwise
            # there will be some double-up
            if nthreads <= ncpus_per_task:
                if not hyperthreading:
                    # The gaps in IDs between assigned cores of sequential OMP threads (should all equal 1)
                    strides = set([j - i for i, j in zip(sorted(cores)[:-1], sorted(cores)[1:])])
                    if ((strides != {1}) or (len(set(cores)) != nthreads)):
                        print('PROC: Close, PLACES: Threads - All OMP threads should be on adjacent hardware threads')
                        return False
                    return True
                else:
                    # Get the CID of each hardware thread, such that both threads on same core will give the same
                    # CID, i.e. CID of 0 and 24 when there are 24 CPUs per socket will both equal 0
                    cores_mod = [cid % ncpus_per_socket for cid in cores]
                    strides_mod = set([j - i for i, j in zip(sorted(cores_mod)[:-1], sorted(cores_mod)[1:])])
                    if ((strides_mod != {0,1}) or (len(set(cores)) != nthreads)):
                        print('PROC: Close, PLACES: Threads - All threads should be on adjacent (physical/logical) threads')
                        return False
                    return True
            else:
                if len(set(cores)) != ncpus_per_task:
                    print('PROC: Close, PLACES: Threads - All available threads should be filled before overloading')
                    return False
                return True
        elif places == 'CORES':
            if nthreads <= ncpus_per_task:
                if not hyperthreading:
                    strides = set([j - i for i, j in zip(sorted(cores)[:-1], sorted(cores)[1:])])
                    if ((strides != {1}) or (len(set(cores)) != nthreads)):
                        print('PROC: Close, PLACES: Cores - All threads should be on adjacent cores')
                        return False
                    return True
                else:
                    cores_mod = [cid % ncpus_per_socket for cid in cores]
                    strides_mod = set([j - i for i, j in zip(sorted(cores_mod)[:-1], sorted(cores_mod)[1:])])
                    # With PLACES=CORES, even with hyperthreading, only one thread will be placed onto each core
                    # until all cores have been occupied, then there will be double-up
                    if ((strides_mod != {1}) or (len(set(cores)) != nthreads)):
                        print('PROC: Close, PLACES: Cores - All threads should be on adjacent cores')
            else:
                if len(set(cores)) != ncpus_per_task:
                    print('PROC: Close, PLACES: Cores - All available threads/cores should be filled before overloading')
        # For CLOSE+SOCKETS there is virtually no restriction on placement
        # NOTE: May change to be a comparison to valid places to be a bit more comprehensive
        elif places == 'SOCKETS':
            if len(set(cores)) != min([nthreads, ncpus_per_task]):
                print('PROC: Close, PLACES: Sockets - OMP threads have not been assigned properly')
                return False
            return True
    # Logic handling for when OMP_PROC_BIND=SPREAD
    # In general, OMP threads should be assigned to cores in a stride such that they are as spread out
    # as they can be. This should continue until all valid cores are filled, and then doubling-up if there
    # are still extra OMP threads
    elif proc_bind == 'SPREAD':
        # The same logic applies for both OMP_PLACES=THREADS and OMP_PLACES=CORES
        if places in ('THREADS', 'CORES'):
            # Same logic for 1 or 2 threads, but the stride between consecutive OMP threads
            # needs to be adjusted for the SMT in the latter case
            if not hyperthreading:
                stride_upp = ceil(ncpus_per_task / nthreads)
                stride_low = floor(ncpus_per_task / nthreads)
                strides = set([j - i for i, j in zip(sorted(cores)[:-1], sorted(cores)[1:])])
            else:
                stride_upp = ceil(ncpus_per_task / ncpus_per_core / nthreads)
                stride_low = floor(ncpus_per_task / ncpus_per_core / nthreads)
                cores_mod = [cid % ncpus_per_socket for cid in cores] # Factor in SMT into TIDs when hyperthreading
                strides = set([j - i for i, j in zip(sorted(cores_mod)[:-1], sorted(cores_mod)[1:])])

            if nthreads <= ncpus_per_task:
                if ((max(strides) > stride_upp) or (min(strides) < stride_low)):
                    print('{PROC: Spread, PLACES: Threads/Cores - OMP threads are not spaced properly')
                    return False
                return True
            else:
                if max(strides) > stride_upp:
                    print('PROC: Spread, PLACES: Threads/Cores - OMP threads are not spaced properly')
                    return False
                return True
        # Virtually no restrictions for SPREAD+SOCKETS
        # Only check to make sure all cores are filled before any doubling-up (if any) occurs
        elif places == 'SOCKETS':
            if len(set(cores)) != min([nthreads, ncpus_per_task]):
                print('PROC: Spread, PLACES: Sockets - OMP threads have not been assigned properly')
                return False
            return True
    # Logic handling for when OMP_PROC_BIND=TRUE and OMP_PLACES is not specified
    # A round-robin approach to assigning OMP threads to cores is used. Threads are assigned
    # to available cores sequentially. If hyperthreading, both threads of a core are available
    # No doubling-up occurs until all cores have been filled
    elif proc_bind == 'TRUE':
        if not hyperthreading:
            strides = set([j - i for i, j in zip(sorted(cores)[-1:], sorted(cores)[1:])])
        else:
            cores_mod = [cid % ncpus_per_socket for cid in cores]
            strides = set([j - i for i, j in zip(sorted(cores_mod)[-1:], sorted(cores_mod)[1:])])

        if len(set(cores)) != min([nthreads, ncpus_per_task]):
            print('PROC: True - All available cores should be filled before overloading')
            return False
        elif max(strides) > 1:
            print('PROC: True - All OMP threads should be on adjacent (logical or physical) CPUs/cores')
            return False
        return True
    # Logic handling for when OMP_PROC_BIND=FALSE (OMP_PLACES need not be specified)
    # A round-robin approach to assigning OMP threads to cores is used. Threads are assigned
    # to available cores randomly. If hyperthreading, both threads of a core are available
    elif proc_bind == 'FALSE':
        if len(set(cores)) != min([nthreads, ncpus_per_task]):
            print('PROC: False - All available cores should be filled before overlaoding')
            return False
        return True
    # The above are all the valid options - this condition refers to an invalid option
    # passed for OMP_PROC_BIND and/or OMP_PLACES
    else:
        print('Invalid combination of OPM_PROC_BIND and OMP_PLACES! Check the test parameters')
        return False


# NOTE: assumes `--ntasks-per-core=1` - if added to tests, need to update logic here
def check_binding(sys_config, job_config, cores):

    # Get relevant fields from dictionaries
    ntasks = job_config['num_tasks_per_node'] # 8
    hyperthreading = job_config['hyperthread'] # TRUE
    nthreads = job_config['omp_num_threads'] # 4
    ncpus_per_task = job_config['num_cpus_per_task'] # 4
    ncpus_per_socket = sys_config['num_cpus_per_socket'] # 128
    nsockets = sys_config['num_sockets'] # 2
    if hyperthreading:
        #ncpus_per_socket = sys_config['num_cpus_per_socket'] # 128
        ncpus_per_core = sys_config['num_cpus_per_core'] # 2
    else:
        #ncpus_per_socket = sys_config['num_cpus_per_socket'] // sys_config['num_cpus_per_core'] # 64
        ncpus_per_core = 1
    ncpus_per_l3 = 8

    # Default behaviour (on Magnus) appears to be to bounce across
    # sockets/NUMA regions (they are the same sets of CPUs)
    # So, for 1 CPU per task, no hyperthreading, we would expect 0 -> 0, 1 -> 12, 2 -> 1, 3 -> 13, etc.
    # With hyperthreading, we would expect 0 -> 0, 1 -> 24, 2 -> 1, 3 -> 25, etc.
#The MPI binding on node nid001178 is:  ['0 -> 0', '1 -> 64', '2 -> 2', '3 -> 66', '4 -> 4', '5 -> 68', '6 -> 6', '7 -> 70'] SUCCESS
    # Strides between cores of consecutive MPI ranks
    diffs = set([abs(j - i) for i, j in zip(cores[:-1], cores[1:])])

    # Valid strides allowed given system and job configuration
    if ncpus_per_task <= ncpus_per_l3:
        valid_diffs = {ncpus_per_l3}
        if ntasks > (ncpus_per_socket // ncpus_per_l3):
            valid_diffs |= {ncpus_per_socket - ncpus_per_l3 - (ncpus_per_task // ncpus_per_core)}
    else:
        valid_diffs = {
            ceil(ncpus_per_task / ncpus_per_l3) * ncpus_per_l3,
            ncpus_per_socket - (ceil(ncpus_per_task / ncpus_per_l3) * ncpus_per_l3) - (ncpus_per_task // ncpus_per_core)}

    # The set of actual strides should be within the set of valid ones
    if diffs | valid_diffs != valid_diffs:
        return False
    return True

def main():

    # Parse command line arguments
    sys_config, job_config, mode = parse_args()

    # Print the system and job step configuration
    printConfig(sys_config, job_config)

    # Get affinity information from file
    nodes, ranks, tids, cores = extract_affinity(job_config)
    # Given inconsistenty of printed output with multiple
    # MPI tasks, sort all three lists by ascending MPI rank
    triples = sorted(zip(nodes, ranks, tids, cores))
    nodes, ranks, tids, cores = [t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples], [t[3] for t in triples]

    num_nodes = job_config['num_nodes']
    ntasks = job_config['num_tasks_per_node']
    nthreads = job_config['omp_num_threads']
    # OMP thread affinity checking
    if mode == 'OMP':
        for inode in range(0, num_nodes * ntasks * nthreads, ntasks * nthreads):
            curr_node = nodes[inode]
            node_ranks = ranks[inode:inode+(ntasks*nthreads)]
            node_tids = tids[inode:inode+(ntasks*nthreads)]
            node_cores = cores[inode:inode+(ntasks*nthreads)]
            # Iterate over each rank, and do the affinity checking for each one
            passfail_list = [None for _ in range(ntasks)]
            for irank in range(0, ntasks * nthreads, nthreads):
                curr_rank = node_ranks[irank]
                print(curr_rank)
                # Display the OMP thread affinity for MPI rank `curr_rank`
                print('THE ASSIGNED THREADS FOR MPI RANK ' + str(curr_rank) + ' ON NODE ' + curr_node + ' -> CPUS: ',
                    ['%s -> %s' % (t[2], t[3]) for t in triples[irank:irank+nthreads]])

                # Boolean to hold whether the reported affinity is suitable
                # ReFrame test searches for the string printed here within the sanity test
                good_affinity = check_affinity(sys_config, job_config,
                                            tids[irank:irank+nthreads], cores[irank:irank+nthreads])
                if good_affinity:
                    print_colour('Affinity for MPI rank %d on node %s is sensible!' % (curr_rank, curr_node), 
                                 colours.GREEN)
                else:
                    print_colour('Affinity for MPI rank %d on node %s is unexpected!' % (curr_rank, curr_node), 
                                 colours.RED)
                passfail_list[irank // nthreads] = good_affinity
            # Check if all ranks have sensible affinity
            # If all ranks have sensible affnity, then the job as a whole does
            if all(passfail_list):
                print_colour('OMP THREAD AFFINITY ON NODE %s IS SENSIBLE!\n' % curr_node, colours.GREEN)
            else:
                print_colour('OMP THREAD AFFINITY ON NODE %s IS UNEXPECTED!\n' % curr_node, colours.RED)
    # MPI rank affinity checking
    elif mode == 'MPI':
        for inode in range(0, num_nodes * ntasks * nthreads, ntasks * nthreads):
            curr_node = nodes[inode]
            node_ranks = ranks[inode:inode+(ntasks*nthreads)]
            node_cores = cores[inode:inode+(ntasks*nthreads)]
            # MPI ranks are bound to core/thread that master OMP thread is on
            master_cids = node_cores[::nthreads]
            print('The MPI binding on node', curr_node, 'is: ', ['%d -> %s' % (node_ranks[j*nthreads], master_cids[j]) for j in range(ntasks)])
            good_binding = check_binding(sys_config, job_config, master_cids)
            if good_binding:
                print_colour('MPI RANK AFFINITY IS SENSIBLE!\n', colours.GREEN)
            else:
                print_colour('MPI RANK AFFINITY IS UNEXPECTED!\n', colours.RED)


if __name__ == "__main__":
    main()