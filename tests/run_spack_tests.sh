#!/bin/bash

repo_root_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
rfm_settings_file=${repo_root_dir}/setup_files/settings.py

# Define help message to be printed
usage() {
    declare -r script_name=$(basename "$0")
    echo """
Usage:
"${script_name}" [option] <name>

Option:
    [-e]: optional list of spack build environments to run the reframe tests over
    [-o]: optional reframe command-line arguments one wishes to use
"""
}

# Default value of paossible passed arguments
reframe_opts=""
env_list=""

# Process command-line arguments
while getopts ':e:o:' opt; do
  case "${opt}" in
    e)
        env_list=${OPTARG}
        ;;
    o)
        # Convert comma-separated string of options to array of option strings
        options=(${OPTARG//,/ })
        # Convert array of option strings to one string of options separated by whitespace ' '
        for i in ${options[@]}
        do
            reframe_opts="${reframe_opts} ${i}"
        done
        echo "Running spack tests with the following reframe options: $reframe_opts"
        ;;
    h | *)
      usage
      exit 0
      ;;
  esac
done
shift $((OPTIND-1))

# Call Reframe according to passed arguments
if [ -z "$env_list" ]
then
    printf "Running spack tests with a single build environemnt\n\n"
    echo reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests/spack/spack_checks.py -r $reframe_opts
else
    printf "Running spack tests for each of the following spack build environments: $env_list\n\n"
    for env in $env_list; do
        export SPACK_ENV=${env}
        printf "Running spack tests for spack build environment $env\n\n"
        echo reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests/spack/spack_checks.py -r $reframe_opts
        unset SPACK_ENV
    done
fi
