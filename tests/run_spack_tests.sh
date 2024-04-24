#!/bin/bash

repo_root_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
rfm_settings_file=${repo_root_dir}/setup_files/settings.py

# All packages are in one spack build environment
if [ -z "$1" ]
then
  printf "Running spack tests with a single spack build environment\n\n"
  reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests/spack/spack_checks.py -l -v
# Packages are split across multiple build environments
else
  env_list=$1
  printf "Running spack tests for each of the following spack build environments: $1\n\n"
  for env in $env_list; do
    export SPACK_ENV=${env}
    printf "Running spack tests for each of the following spack build environments: $env\n\n"
    reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests/spack/spack_checks.py -l -v
    unset SPACK_ENV
  done
fi
