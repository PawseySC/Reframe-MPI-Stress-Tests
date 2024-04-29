#!/bin/bash

# Define help message to be printed
usage() {
    declare -r script_name=$(basename "$0")
    echo """
Usage:
"${script_name}" [option] <name>

Option:
    [-f]: Specify a name of a test file and run all tests within that file
    [-t]: Specify a category/tag and run all tests with that tag
    [-n]: Specify the name of a single test - or multiple tests each separated by | - and run the test(s)
    [-a]: Run all tests 
    [-o]: optional reframe command-line arguments one wishes to use
"""
}

mode=""
value=""

# Process command-line arguments
while getopts ':f:c:n:ae:o:' opt; do
  case "${opt}" in
    f)
      echo "optarg = ${OPTARG}"
      mode="test file"
      value=${OPTARG}
      ;;
    t)
      mode="test category"
      value=${OPTARG}
      ;;
    n)
      if [[ $OPTARG == *"|"* ]]; then
        mode="multiple tests"
      else
        mode="single test"
      fi
      value=${OPTARG}
      ;;
    a)
      mode="all tests"
      ;;
    o)
      # Convert comma-separated string of options to array of option strings
      options=(${OPTARG//,/ })
      # Convert array of option strings to one string of options separated by whitespace ' '
      for i in ${options[@]}
      do
          reframe_opts="${reframe_opts} ${i}"
      done
      echo "Running tests with the following reframe options: $reframe_opts"
      ;;
    h | *)
      echo "Usage"
      usage
      exit 0
      ;;
  esac
done
shift $((OPTIND-1))

# If there is no specified mode, exit with help message
if [ -z "${mode}" ]; then
    printf "At least one of -f, -t, -n, or -a must be specified\n\n"
    usage
    exit 0
fi

# Settings file needed for Reframe
repo_root_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
rfm_settings_file=${repo_root_dir}/setup_files/settings.py

# Call Reframe based on passed arguments
# Exclude spack related tags via -T option since they have a separate dedicated script to run
if [[ "$mode" == "test file" ]]; then
  printf "Running tests from test file $value \n\n"
  echo reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests/$value -T spack -r --performance-report $reframe_opts
elif [[ "$mode" == "test category" ]]; then
  printf "Running tests from test category $value \n\n"
  echo reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -t $value -T spack -r --performance-report $reframe_opts
elif [[ "$mode" == "single test" ]]; then
  printf "Running test $value \n\n"
  echo reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -n $value -T spack -r --performance-report $reframe_opts
elif [[ "$mode" == "multiple tests" ]]; then
  printf "Running tests $value \n\n"
  echo reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -n "$value" -T spack -r --performance-report $reframe_opts
elif [[ "$mode" == "all tests" ]]; then
  printf "Running all tests\n\n"
  echo reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -T spack -r --performance-report $reframe_opts
fi
