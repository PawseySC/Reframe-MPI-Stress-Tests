import re
import sys
import os.path

# Add root directory of repo to path
curr_dir = os.path.dirname(__file__).replace('\\','/')
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)


# Populate lists with entries from test environment config file
def add_to_env(lines, env_str, mod_str, cmd_str, env_vars, modules, cmds):
    env_patt = rf'^{env_str}:\s(\S+)'
    mod_patt = rf'^{mod_str}:\s(\S+)'
    cmd_patt = rf'^{cmd_str}:\s(.*)'
    for line in lines:
        env_match = re.match(env_patt, line)
        mod_match = re.match(mod_patt, line)
        cmd_match = re.match(cmd_patt, line)
        if env_match != None:
            env_vars += [env_match.groups()[0]]
        elif mod_match != None:
            modules += [mod_match.groups()[0]]
        elif cmd_match != None:
            cmds += [cmd_match.groups()[0]]
        else:
            continue

# Default environment variables are for a Cray-MPICH system
def set_env(mpi = True, omp = False, sched = False, gpu = False):

    # Memory and environment reporting variables for Cray-MPICH
    # Edit file if needed for your site/setup
    config_file = root_dir + '/setup_files/test_env.config'
    # Lists to hold env vars, modules, and commands
    env_vars = []
    modules = []
    cmds = []

    f = open(config_file, 'r')
    lines = f.readlines()
    # Filter for MPI, OMP, GPU, and job scheduler related entries
    if mpi:
        add_to_env(lines, 'mpi-env', 'mpi-mod', 'mpi-cmd', env_vars, modules, cmds)
    if omp:
        add_to_env(lines, 'omp-env', 'omp-mod', 'omp-cmd', env_vars, modules, cmds)
    if gpu:
        add_to_env(lines, 'gpu-env', 'gpu-mod', 'gpu-cmd', env_vars, modules, cmds)
    if sched:
        add_to_env(lines, 'sched-env', 'sched-mod', 'sched-cmd', env_vars, modules, cmds)
    # Filter for module system realted commands
    for line in lines:
        match = re.match(r'^mod-cmd:\s(.*)', line)
        if match == None:
            continue
        cmds += [match.groups()[0]]
    f.close()

    return env_vars, modules, cmds
