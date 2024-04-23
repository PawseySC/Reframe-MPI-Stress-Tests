import yaml

def load_yaml(filepath):

    with open(filepath, 'r') as stream:
        data = yaml.safe_load(stream)

    return data


def set_system(filepath, testname = None):

    data = load_yaml(filepath)
    
    if testname is not None:
        sys_info = data[testname]['system-parameters']
    else:
        sys_info = data['system-parameters']

    return sys_info


def set_env(filepath):

    data = load_yaml(filepath)

    env_info = data['environment']

    modules = []
    cmds = []
    env_vars = {}

    for key, value in env_info.items():
        # if value == False:
        #     continue
        # else:
        new_modules = value['modules']
        new_commands = value['commands']
        new_vars = value['env-vars']
        if new_modules is not None:
            modules += new_modules
        if new_commands is not None:
            cmds += new_commands
        if new_vars is not None:
            env_vars.update(value['env-vars'])

    return env_vars, modules, cmds


def configure_test(filepath, testname):

    data = load_yaml(filepath)

    test_info = data[testname]
    if 'test-parameter' in test_info.keys():
        del test_info['test-parameter']

    return test_info

def get_job_options(filepath, testname = None):

    data = load_yaml(filepath)

    if testname is not None:
        job_opts = data[testname]['job-options']
    else:
        job_opts = data['job-options']

    return job_opts

def get_test_params(filepath, testname = None):

    data = load_yaml(filepath)
    if testname is not None:
        test_info = data[testname]
        return test_info['test-parameters']
    else:
        return data['test-parameters']