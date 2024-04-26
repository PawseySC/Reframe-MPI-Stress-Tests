import re
import os
import sys
import yaml
import json

# Add root directory of repo to path
curr_dir = os.path.dirname(__file__).replace('\\','/')
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
root_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(root_dir)
# Import functions to set env vars, modules, commands
from common.scripts.parse_yaml import *
config_path = curr_dir + '/spack_config.yaml'

# Get spack info (directory of spack.yaml and spack.lock files and path of modules.yaml file)
def get_spack_info(filepath):

    data = load_yaml(filepath)
    spack_info = data['spack-setup']
    return spack_info

# Get format of full path to installed software/libraries
def get_sw_path_format(filepath):

    data = load_yaml(filepath)
    lib_dict = data['software-path']
    return lib_dict

# Get format of full path to module file
def get_module_path_format(filepath):

    data = load_yaml(filepath)
    mod_path_dict = data['module-path']
    return mod_path_dict

# Get basic sanity commands to use in baseline sanity check
def get_pkg_cmds(filepath):

    data = load_yaml(filepath)
    pkg_cmds = data['commands']
    return pkg_cmds

# Build software or module path for a package
# path_dict is from spack_config.yaml file and pkg_spec is the concretised spec of the package
def set_path(path_dict, pkg_spec):

    path = ''
    # Iterate through every entry in the dictionary, which correspond to a level of the directory hierarchy
    for key, val in path_dict.items():
        # A fixed directory name for this level, independent of the specific package
        if ('{' not in val) and ('}' not in val):
            path += (val + '/')
        # The directory name at this level is variable for different packages
        else:
            # If multiple entries from the concrete specs are part of this level's name
            # In yaml file different entries in the spec for the same directory level are separated by '-'
            if '-' in val:
                # Handle each part separately
                parts = val.split('-')
                nparts = len(parts)
                for i in range(nparts):
                    # Go deeper into the spec dictionary until we are at the relevant info
                    # In yaml file, different levels in the dictionary are separated by ','
                    segment = pkg_spec
                    entries = parts[i].replace('{', '').replace('}', '').split(',')
                    for entry in entries:
                        segment = segment[entry]
                    # Add '/' for final part of this level, otherwise '-'
                    if i == nparts - 1:
                        path += (segment + '/')
                    else:
                        path += (segment + '-')
            # A single entry from the spec constitutes the name of the directory at this level
            else:
                entries = val.replace('{', '').replace('}', '').split(',')
                segment = pkg_spec
                for entry in entries:
                    segment = segment[entry]
                path += (segment + '/')

    return path

# Get the full set of abstract specs for an environemtn from the spack.yaml file
def get_abstract_specs():

    spack_dict = get_spack_info(config_path)
    yaml_dir = spack_dict['spack-yaml-dir']

    # Abstract specs we wish to install
    abstract_specs = []

    if os.getenv('SPACK_ENV') is not None:
        env = os.getenv('SPACK_ENV')
        #yaml_file = curr_dir + f'/src/environments/{env}/spack.yaml'
        yaml_file = yaml_dir + f'/{env}/spack.yaml'
    else:
        #yaml_file = curr_dir + '/src/spack.yaml'
        yaml_file = yaml_dir + '/spack.yaml'
    with open(yaml_file, "r") as stream:
        data = yaml.safe_load(stream)

    # The spec categories (in the matrices in the spack.yaml file)
    spec_categories = []
    for entry in data['spack']['specs']:
        m = entry['matrix']
        # Only add those categories which start with '$', which are expanded out
        if m[0][0][0] == '$':
            spec_categories.append(m[0][0][1:])

    for entry in data['spack']['definitions']:
        # Iterate over each group of packages
        for key, value in entry.items():
            # Select only those categories which are listed in the matrices
            if key in spec_categories:
                # Add each individual packages spec in this group
                for elem in value:
                    abstract_specs.append(elem)

    # Handle cases where specs are defined within a matrix rather than with other packages (e.g. cmake in utils env)
    for entry in data['spack']['specs']:
        for key, value in entry.items():
            if value[0][0][0] != '$':
                abstract_specs.append(value[0][0])
    
    # Sort the list and return
    return sorted(abstract_specs)

# Get spec (plus full hash) for each root package in an environment from spack.lock file
def get_root_specs():

    spack_dict = get_spack_info(config_path)
    json_dir = spack_dict['spack-lock-dir']

    # Fully concretised spacks generated from concretisation
    root_specs = []

    if os.getenv('SPACK_ENV') is not None:
        env = os.getenv('SPACK_ENV')
        #json_file = curr_dir + f'/src/environments/{env}/spack.lock'
        json_file = json_dir + f'/{env}/spack.lock'
    else:
        #json_file = curr_dir + '/src/spack.lock'
        json_file = json_dir + f'/spack.lock'
    with open(json_file) as json_data:
        data = json.load(json_data)
    
    # Iterate over every package
    for entry in data['roots']:
        # Get full spec and hash for each package
        s = entry['spec']
        h = entry['hash']
        root_specs.append(s + ' ' + h)

    # Sort the list and return
    return sorted(root_specs)

# Get full concretised spec (plus full hash) for each spec from spack.lock file
def get_concretised_specs():

    spack_dict = get_spack_info(config_path)
    json_dir = spack_dict['spack-lock-dir']

    # Fully concretised spacks generated from concretisation
    if os.getenv('SPACK_ENV') is not None:
        env = os.getenv('SPACK_ENV')
        #json_file = curr_dir + f'/src/environments/{env}/spack.lock'
        json_file = json_dir + f'/{env}/spack.lock'
    else:
        #json_file = curr_dir + '/src/spack.lock'
        json_file = json_dir + '/spack.lock'
    with open(json_file) as json_data:
        data = json.load(json_data)
    concretised_specs = data['concrete_specs']

    # Sort the list and return
    return concretised_specs


# Get path to the shared object libraries for a package
def get_software_path(pkg_name_ver):

    # Get format of library paths from yaml config file
    sw_dict = get_sw_path_format(config_path)

    # Get root and concretised specs for this environment
    root_specs = get_root_specs()
    conc_specs = get_concretised_specs()

    # Iterate over every spec
    pkg_name = pkg_name_ver[0]
    pkg_ver = pkg_name_ver[1][:-4]
    # Handle packages with specifiers after version in module file (e.g. of the form petsc/3.19.5-complex)
    # log4cxx is treated separately since it has package version and C++ version in module file
    if '-' in pkg_ver: 
        pkg_ver = pkg_ver.split('-')[0]
    
    for idx, s in enumerate(root_specs):
        # Hash is last entry in spec
        h = s.split(' ')[-1]
        c = conc_specs[h]
        name = c['name']
        ver = c['version']
        if (name == pkg_name) and (ver == pkg_ver):
            sw_path = set_path(sw_dict, c)
            break

    return sw_path

# Build a spec of the format of those specified in the `roots` section of the spack.lock file
def build_root_spec(conc_spec, param_dict):

    # Get package information from the full concrete spec
    name = conc_spec['name']
    ver = conc_spec['version']
    h = conc_spec['hash']
    comp_name = conc_spec['compiler']['name']
    comp_ver = conc_spec['compiler']['version']

    # Start with the basic components of the spec
    root_spec = name + '@' + ver + '%' + comp_name + '@' + comp_ver

    # Add to spec from the parameters dictionary of the concrete spec
    # Only add the True/False parameter values in the format of +param and ~param
    for key, val in param_dict.items():
        if val == False:
            root_spec += ('~' + key)
        elif val == True:
            root_spec += ('+' + key)
        elif isinstance(val, list):
            continue
        
    # Add key-value parameters from parameters dictionary of the
    # concrete spec in the format key=value
    for key, val in param_dict.items():
        if isinstance(val, list):
            continue
        if (val != False) and (val != True):
            root_spec += (' ' + key + '=' + val)
    
    # Add the hash to the end of the spec
    root_spec += (' ' + h)

    return root_spec

# Get full module path for this package
# NOTE: This is used primarily for dependencies, which are not root packages of any of the environments
def get_dependency_module_path(pkg_info, conc_specs, root_spec):

    # Get required environment variables
    spack_dict = get_spack_info(config_path)
    python_ver = spack_dict['python-version']
    mod_yaml_file = spack_dict['module-yaml-path']

    # Master module file describing format of full module path across all environments
    with open(mod_yaml_file, "r") as stream:
        mod_data = yaml.safe_load(stream) 
    # Projections describe module paths for each package
    projections = mod_data['modules']['default']['lmod']['projections']
    projs = []
    paths = []
    # Fill lists with corresponding projections and paths
    for proj in projections:
        paths.append(projections[proj])
        projs.append(proj)

    # Details of dependency
    name = pkg_info['name']
    h = pkg_info['hash']
    version  = pkg_info['version']
    comp = pkg_info['compiler']['name'] + '/' + pkg_info['compiler']['version']
    arch = pkg_info['arch']['target']['name']

    mod_path_dict = get_module_path_format(config_path)
    base_mod_path = set_path(mod_path_dict, conc_specs[h])

    # Get index of this package in the set of all concretised specs by matching hashes
    hashes = [c for c in conc_specs]
    idx = [i for i, j in enumerate(hashes) if j == h][0]

    # Find mathching projection for the spec of this package
    matching_projections = []
    for p_idx, p in enumerate(projs):
        # Split projection into list of specifiers
        specifiers = p.split(' ')
        if name in p:
            mask = ['~' not in s for s in specifiers]
            # Iterate through all variants, updating if needed
            updated_specifiers = [b for a, b in zip(mask, specifiers) if a]
            # If all specifiers of a variant are present in the package spec, it matches
            if all(s in root_spec for s in updated_specifiers):
                matching_projections.append(p)

    # Handle dependencies which are a partial match to a projection, but not a full match
    # These are then hidden module files in the dependencies directory
    # One example is boost/1.80.0 dependency for hpx/1.8.1 - there are explicit boost projections, 
    # but this particular boost variant does not have a matching projection,
    # so it goes under dependencies catch-all projection
    if len(matching_projections) == 0:
        full_mod_path = base_mod_path + paths[-1].replace('{name}', name).replace('{version}', version).replace('{hash:7}', h[:7]) + '.lua'
        return full_mod_path
        

    # There is more than one possible matching projection
    if len(matching_projections) > 1:
        nmatches = []
        for idx, proj in enumerate(matching_projections):
            # Split projection into it's components (i.e. "gromacs +double" -> ['gromacs', '+double'])
            tmp = proj.split(' ')
            # Number of keywords in this projection which are found in the package concretised spec
            nmatches.append(sum([keyword in root_spec for keyword in tmp]))
        # Pick the projection which has the most keyword matches
        max_idx = max(enumerate(nmatches), key=lambda x: x[1])[0]
        matched_proj = matching_projections[max_idx]
    # There is only one matched projection
    else:
        matched_proj = matching_projections[0]


    # Get module path for the matched projection
    for p_idx, p in enumerate(projs):
        if p == matched_proj:
            # Do we need to include the python version (e.g. for mpi4py)?
            if '^python.version' in paths[p_idx]:
                matching_mod_path = paths[p_idx].replace('{name}', name).replace('{version}', version).replace('{^python.version}', python_ver)
            # Standard projection
            else:
                matching_mod_path = paths[p_idx].replace('{name}', name).replace('{version}', version)

    # Convert the matching module path into full absolute path
    full_mod_path = base_mod_path + matching_mod_path + '.lua'

    return full_mod_path

# Get the full module paths for all dependencies for this package
def get_module_dependencies(pkg_module_path):

    # Get required environment variables
    spack_dict = get_spack_info(config_path)
    python_ver = spack_dict['python-version']
    mod_yaml_file = spack_dict['module-yaml-path']

    # Get root specs from the environment this package is in
    root_specs = get_root_specs()
    conc_specs = get_concretised_specs()

    # Master module file describing format of full module path across all environments
    with open(mod_yaml_file, "r") as stream:
        mod_data = yaml.safe_load(stream)    
    # Projections describe module paths for each spec
    projections = mod_data['modules']['default']['lmod']['projections']
    projs = []
    paths = []
    # Fill lists with corresponding projections and paths
    for proj in projections:
        paths.append(projections[proj])
        projs.append(proj)
    
    # List to hold full absolute module paths for every dependency of this package
    dep_paths = []

    # Get relevant package info (name, ver, compiler, architecture) from full module path
    pkg_info = pkg_module_path[:-4].split('/')[-6:]
    if '-' in pkg_info[-1]: # Handle those modules with info between version and .lua file extension
        pkg_info = pkg_info[:-1] + [pkg_info[-1].split('-')[0]]

    mod_path_dict = get_module_path_format(config_path)

    # Iterate over every root spec across all environments
    for idx, s in enumerate(root_specs):
        # Hash is last entry in spec
        h = s.split(' ')[-1]
        # Get full concrete specs for this root spec
        c = conc_specs[h]
        name = c['name']
        ver = c['version']
        comp_name = c['compiler']['name']
        comp_ver = c['compiler']['version']
        arch = c['arch']['target']['name']
        # This root spec is a full match for this module
        if all(map(lambda v: v in pkg_info, [name, ver, comp_name, comp_ver, arch])):
            # Check if it has any dependencies
            key_list = [k for k in c.keys()]
            if 'dependencies' in key_list:
                deps = c['dependencies']
                # Iterate through dependencies
                for d in deps:
                    dh = d['hash']
                    dc = conc_specs[dh]
                    d_comp = dc['compiler']['name'] + '/' + dc['compiler']['version']
                    d_arch = dc['arch']['target']['name']
                    # See if any projections are at least a partial match to this dependency
                    nchars = len(dc['name'])
                    proj_matches = [dc['name'] == proj[0:nchars] for proj in projs]
                    if any(proj_matches):
                        rs = [r for r in root_specs if dh in r]
                        # Build a "root spec" for this dependency and get its full module path
                        built_spec = build_root_spec(dc, dc['parameters'])
                        path = get_dependency_module_path(dc, conc_specs, built_spec)
                    # There is not even a partial projection match, so it falls under the dependencies catch-all projection
                    else:
                        base_mod_path = set_path(mod_path_dict, c)
                        path = base_mod_path + paths[-1].replace('{name}', dc['name']).replace('{version}', dc['version']).replace('{hash:7}', dh[:7]) + '.lua'
                    dep_paths.append(path)
    
    return dep_paths

# Get full absolute module paths for every package in an environment
def get_module_paths():

    # Get required environment variables
    spack_dict = get_spack_info(config_path)
    python_ver = spack_dict['python-version'] # Some python packages include python version in their name (e.g. mpi4py)
    mod_yaml_file = spack_dict['module-yaml-path']

    # Get root specs for this environment (from environments/{env}/spack.lock file)
    # Three lists - full concretised spec, reduced list in format {name}/{version}, and hash
    root_specs = get_root_specs()
    root_name_ver = [None] * len(root_specs)
    hashes = [None] * len(root_specs)

    #########################################
    # Convert full spec to name and version #
    #########################################
    conc_specs = get_concretised_specs()

    mod_path_dict = get_module_path_format(config_path)
    mod_paths = [''] * len(root_specs)

    # Given root spec is not in universally consistent format, use full concrete specs to get compilers, architecture, etc.
    for idx, s in enumerate(root_specs):
        # Hash is last entry in spec
        h = s.split(' ')[-1]
        hashes[idx] = h
        c = conc_specs[h]
        mod_paths[idx] = set_path(mod_path_dict, c)
        root_name_ver[idx] = c['name'] + '/' + c['version']

    ##################################################
    # Matching concretised specs to full module path #
    ##################################################
    # Master module file describing format of full module path across all environments
    with open(mod_yaml_file, "r") as stream:
        mod_data = yaml.safe_load(stream)    
    # Projections describe module paths for each spec
    projections = mod_data['modules']['default']['lmod']['projections']
    projs = []
    paths = []
    # Fill lists with corresponding projections and paths
    for proj in projections:
        paths.append(projections[proj])
        projs.append(proj)

    # Match each projection with the corresponding abstract spec
    matching_projections = [ [] for _ in range(len(root_specs))]
    for idx, spec in enumerate(root_specs):
        # Extract name of module from {name}/{version} entries
        name = root_name_ver[idx].split('/')[0]
        # Iterate through each projection
        for p_idx, p in enumerate(projs):
            # If there are variants, extra specifications will be separated from name by ' '
            # Examples: gromacs vs. gromacs +double, lammps ~rocm vs. lammps +rocm
            specifiers = p.split(' ')
            # If the name of this concretised spec is within this projection
            if name in p:
                # Filter variants, removing entries with `~` since it denotes the lack of something
                mask = ['~' not in s for s in specifiers]
                # Iterate through all specifiers, updating if needed
                updated_specifiers = [b for a, b in zip(mask, specifiers) if a]
                # If all specifiers of a projection are present in the spec, it matches
                if all(s in spec for s in updated_specifiers):
                    matching_projections[idx].append(p)

    # Multiple projections can match a single spec, we need to pick the one true match
    # Example: A spec with "gromacs +double" can be matched by both "gromacs" and "gromacs +double" projections
    for m_idx, m in enumerate(matching_projections):
        # If there is more than one matching projection for this spec
        if len(m) > 1:
            # List to hold the number of keywords in each possible match that are found in the concretised spec
            # The possible match with the highest number of keyword matches is selected as the correct match
            # Example: "gromacs +double" is matched with both "gromacs" and "gromacs +double" projections
            # so "gromacs +double" has 2 matches and "gromacs" just 1, so "gromacs +double" projection is chosen
            nmatches = []
            for idx, proj in enumerate(m):
                # Split projection into it's components (i.e. "gromacs +double" -> ['gromacs', '+double'])
                tmp = proj.split(' ')
                # Number of keywords in this variant which are found in the full concretised spec
                nmatches.append(sum([keyword in root_specs[m_idx] for keyword in tmp]))
            # Find projectionn with highest number of keyword matches
            max_idx = max(enumerate(nmatches), key=lambda x: x[1])[0]
            matching_projections[m_idx] = [m[max_idx]]
    # Convert from list of lists to list of strings
    matching_projections = [m[0] for m in matching_projections if len(m) > 0]

    # List to hold module paths for matched projection of each abstract spec
    matching_mod_paths = [None for _ in range(len(matching_projections))]
    for m_idx, m in enumerate(matching_projections):
        # Name and version are both included
        if '/' in root_name_ver[m_idx]:
            name, ver = root_name_ver[m_idx].split('/')
        # No version information in the spec, need to search spec dictionary entry in .lock file
        else:
            name = root_name_ver[m_idx]
            ver = conc_specs[hashes[m_idx]]['version']
        for p_idx, p in enumerate(projs):
            # This projection matches one of the abstract specs
            if p == m:
                # Do we need to include the python version (e.g. for mpi4py)?
                if '^python.version' in paths[p_idx]:
                    matching_mod_paths[m_idx] = paths[p_idx].replace('{name}', name).replace('{version}', ver).replace('{^python.version}', python_ver)
                # Standard projection
                else:
                    matching_mod_paths[m_idx] = paths[p_idx].replace('{name}', name).replace('{version}', ver)

    # Convert the matching module paths into full absolute paths
    full_mod_paths = [None] * len(matching_mod_paths)
    for i in range(len(matching_mod_paths)):
        full_mod_paths[i] = mod_paths[i] + matching_mod_paths[i] + '.lua'

    return full_mod_paths
