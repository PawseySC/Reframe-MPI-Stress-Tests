import argparse
import re # regex

# Function that use `argparse` module to parse command line arguments
# passed to this script as postrun_cmds in the parent ReFrame test
def parse_args():

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ntasks', help = 'Number of MPI tasks per node')
    parser.add_argument('-N', '--nodes', help = 'Number of nodes')
    parser.add_argument('-f', '--mfile', help = 'File holding the memory reports')
    opts = parser.parse_args()

    # Check all (mandatory) arguments are present
    if not (opts.ntasks and opts.nodes and opts.mfile):
        parser.error('-n, -N, and -f must be provided')

    # Preprocess the arguments before returning them to calling function
    ntasks = int(opts.ntasks)
    num_nodes = int(opts.nodes)

    return ntasks, num_nodes, opts.mfile


def extract_mem_stats(file):

    # Open and red entire contents of file
    with open(file, 'r') as f:
        dump = f.read()

    # Extract system memory reports from within MPI comms program
    # Fields are (node, function, Total, Used, Free, Shared, Cache, Avail) - final 6 include value + units
    system_func_matches = re.findall(
        r'.*(nid[0-9]+).+@\s(\w+).+\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];' +
        r'\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)' +
        r'\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];', dump, re.M)
    nfunc_reports = len(system_func_matches)

    # Extract system memory reports from outside MPI comms program (these are periodic)
    # Fields are (node, Total, Used, Free, Shared, Cache, Avail) - final 6 include value + units
    system_mem_matches = re.findall(
        r'PERIODIC.+(nid[0-9]+).+\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)' +
        r'\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)' +
        r'\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];\s\w+\s*:\s([0-9]+.[0-9]+)\s\[(\w+)\];', dump, re.M)
    nreports = len(system_mem_matches)

    # Fill dictionary with all of the system memory reports
    system_mem_dict = {
        'Node': ["" for _ in range(nreports)],
        'Function': ["" for _ in range(nreports)],
        'Total': [[] for _ in range(nreports)],
        'Used': [[] for _ in range(nreports)],
        'Free': [[] for _ in range(nreports)],
        'Shared': [[] for _ in range(nreports)],
        'Cache': [[] for _ in range(nreports)],
        'Available': [[] for _ in range(nreports)],
    }
    # Separate list for function names since the periodic reports don't
    # include function name in their reports
    functions_list = ["" for _ in range(nreports)]
    current_function = ""
    offset = 0
    for ireport in range(nfunc_reports):
        if system_func_matches[ireport][1] != "main":
            curr_func = system_func_matches[ireport][1]
            offset += 1
            continue
        idx = ireport - offset
        system_mem_dict['Node'][idx] = system_mem_matches[idx][0]
        system_mem_dict['Function'][idx] = curr_func
        system_mem_dict['Total'][idx] = [float(system_mem_matches[idx][1]), system_mem_matches[idx][2]]
        system_mem_dict['Used'][idx] = [float(system_mem_matches[idx][3]), system_mem_matches[idx][4]]
        system_mem_dict['Free'][idx] = [float(system_mem_matches[idx][5]), system_mem_matches[idx][6]]
        system_mem_dict['Shared'][idx] = [float(system_mem_matches[idx][7]), system_mem_matches[idx][8]]
        system_mem_dict['Cache'][idx] = [float(system_mem_matches[idx][9]), system_mem_matches[idx][10]]
        system_mem_dict['Available'][idx] = [float(system_mem_matches[idx][11]), system_mem_matches[idx][12]]



    # Extract process memory reports within MPI comms program
    # Fields are (node, function, VM current, VM peak, RSS current, RSS peak) - last 4 include value + units
    mem_func_matches = re.findall(
        r'.*(nid[0-9]+).+@\s(\w+).+\s\w+\/\w+:\s([0-9]+.[0-9]+)\s\[(\w+)\]\s\/\s([0-9]+.[0-9]+)\s\[(\w+)\];' +
        r'.+\s([0-9]+.[0-9]+)\s\[(\w+)\]\s\/\s([0-9]+.[0-9]+)\s\[(\w+)\]', dump, re.M)
    # Extract process memory reports from outside MPI comms program (these are periodic)
    # Fields are (node, VM current, VM peak, RSS current, RSS peak) - last 4 include value + units
    mem_usage_matches = re.findall(
        r'PERIODIC.+(nid[0-9]+).+\s\w+\/\w+:\s([0-9]+.[0-9]+)\s\[(\w+)\]\s\/\s([0-9]+.[0-9]+)\s\[(\w+)\];' +
        r'.+\s([0-9]+.[0-9]+)\s\[(\w+)\]\s\/\s([0-9]+.[0-9]+)\s\[(\w+)\]', dump, re.M)
    nfunc_reports = len(mem_func_matches)
    nreports = len(mem_usage_matches)


    # Fill dictionary with all the proc mem usage reports
    mem_usage_dict = {
        'Node': ["" for _ in range(nreports)],
        'Function': ["" for _ in range(nreports)],
        'VM_Current': [[] for _ in range(nreports)],
        'VM_Peak': [[] for _ in range(nreports)],
        'RSS_Current': [[] for _ in range(nreports)],
        'RSS_Peak': [[] for _ in range(nreports)],
    }
    # Separate handling of functions since they are only in one of the report sets
    current_function = ""
    offset = 0
    for ireport in range(nfunc_reports):
        if mem_func_matches[ireport][1] != "main":
            curr_func = mem_func_matches[ireport][1]
            offset += 1
            continue
        idx = ireport - offset
        mem_usage_dict['Node'][idx] = mem_usage_matches[idx][0]
        mem_usage_dict['Function'][idx] = curr_func
        mem_usage_dict['VM_Current'][idx] = [float(mem_usage_matches[idx][1]), mem_usage_matches[idx][2]]
        mem_usage_dict['VM_Peak'][idx] = [float(mem_usage_matches[idx][3]), mem_usage_matches[idx][4]]
        mem_usage_dict['RSS_Current'][idx] = [float(mem_usage_matches[idx][5]), mem_usage_matches[idx][6]]
        mem_usage_dict['RSS_Peak'][idx] = [float(mem_usage_matches[idx][7]), mem_usage_matches[idx][8]]

    return system_mem_dict, mem_usage_dict


# Convert all memory reports to units of [GiB]
def conv_factor(unit):

    if unit == 'MiB':
        return 1 / 1024
    elif unit == 'GiB':
        return 1


def main():

    # Parse command line arguments into a dictionary
    # Dictionary contents shown in parse_args() function
    ntasks, num_nodes, mem_file = parse_args()

    # Extract memory usage from memory report log file
    system_mem_dict, mem_usage_dict = extract_mem_stats(mem_file)
    nsys_reports = len(system_mem_dict['Used'])
    nproc_reports = len(mem_usage_dict['VM_Current'])

    # Number of process memory reports per system memory report
    step = (ntasks - num_nodes) // num_nodes
    nodes_list = list(set(system_mem_dict['Node']))

    # Perform summation of process usage on a node-by-node basis
    for node in nodes_list:
        # All memory reports for this node
        sys_node_idxs = [i for i in range(nsys_reports) if system_mem_dict['Node'][i] == node]
        mem_node_idxs = [i for i in range(nproc_reports) if mem_usage_dict['Node'][i] == node]
        nsys = len(sys_node_idxs)
        nmem = len(mem_node_idxs)

        # Iterate through all the memory reports on this node
        for i in range(0, nmem, step):
            sys_idx = i // step
            # Ensure the units of memory are in [GiB]
            sys_used = system_mem_dict['Used'][sys_node_idxs[sys_idx]][0] * conv_factor(
                system_mem_dict['Used'][sys_node_idxs[sys_idx]][1])
            curr_func = system_mem_dict['Function'][sys_node_idxs[sys_idx]]
            # Add up memory used by all procs (converting to [GiB] as necessary)
            proc_used = 0
            for j in range(i, i + step):
                proc_used += mem_usage_dict['RSS_Current'][mem_node_idxs[j]][0] * conv_factor(
                    mem_usage_dict['RSS_Current'][mem_node_idxs[j]][1])
            # Print message for ReFrame to find in sanity function
            print("ON NODE %s >> IN FUNCTION %s >> Memory used by processes: %.3f >> System memory used: %.3f" %
                  (node, curr_func, proc_used, sys_used))
        # Separator between messages for each node
        print("**************************************************")

if __name__ == "__main__":
    main()