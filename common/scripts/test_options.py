import re

def set_job_opts(options):

    job_opts = []

    for key, value in options.items():
        prefix = value[0]
        sep = value[1]
        val = value[2]
        job_opts += [f'{prefix}{key}{sep}{val}']
    
    return job_opts


def add_to_env(lines, env_str, mod_str, env_vars, modules):
    env_patt = rf'^{env_str}:\s(\S+)'
    mod_patt = rf'^{mod_str}:\s(\S+)'
    for line in lines:
        env_match = re.match(env_patt, line)
        mod_match = re.match(mod_patt, line)
        if env_match != None:
            env_vars += [env_match.groups()[0]]
        elif mod_match != None:
            modules += [mod_match.groups()[0]]
        else:
            continue

# Default environment variables are for a Cray-MPICH system
def set_env(mpi = True, omp = False, sched = False, gpu = False):

    # Memory and environment reporting variables for Cray-MPICH
    config_file = 'common/scripts/test_env.config'
    env_vars = []
    modules = []

    f = open(config_file, 'r')
    lines = f.readlines()
    if mpi:
        add_to_env(lines, 'mpi-env', 'mpi-mod', env_vars, modules)
    if omp:
        add_to_env(lines, 'omp-env', 'omp-mod', env_vars, modules)
    if gpu:
        add_to_env(lines, 'gpu-env', 'gpu-mod', env_vars, modules)
    if sched:
        add_to_env(lines, 'sched-env', 'sched-mod', env_vars, modules)
    f.close()

    return env_vars, modules