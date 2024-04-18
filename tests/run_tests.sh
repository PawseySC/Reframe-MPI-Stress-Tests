#!/bin/bash

# Define help message to be printed
usage() {
    declare -r script_name=$(basename "$0")
    echo """
Usage:
"${script_name}" [option] <name>

Option:
    --test-file, --test-category, --test-name, --test-names < file, category, name, or names of subset of tests one wishes to run >
    --options < optional reframe command-line arguments one wishes to use >
"""
}

# --------------- #
# PARSE ARGUMENTS #
# --------------- #

if [ $# -eq 0 ] # No arguments passed, so run all tests
then
  mode="all tests"
  echo "Running all tests"
elif [ $# -eq 1 ] # Script is called with help
then
  case $1 in
    -h | --help)
        usage
        exit 0
        ;;
  esac
elif [ $# -gt 1 ] # At least one (non-help) argument passed
then
  # Handle the first argument (subset of tests to run)
  case $1 in
    --test-file) mode="test file";;
    --test-category) mode="test category";;
    --test-name) mode="test name";;
    --test-names) mode="multiple tests";;
  esac
  # Store the file, category, or test name(s)
  val=$2
  # Shift to move to second argument (options) if it is passed
  shift 2

  # Handle the secound argument (optional Reframe command line arguments to use)
  reframe_opts=""
  if [ -z "$1" ]
  then
    printf "Running with no additional reframe options\n"
  else
    # USe getopt to allow long or short argument name
    VALID_ARGS=$(getopt -o o: --long options: -- "$@")
    eval set -- "$VALID_ARGS"
    while [ : ]; do
      case "$1" in
        -o | --options)
            # Convert comma-separated string of options to array of option strings
            options=(${2//,/ })
            # Convert array of option strings to one string of options separated by whitespace ' '
            for i in ${options[@]}
            do
              reframe_opts="${reframe_opts} ${i}"
            done
            shift 2
            ;;
        --) shift;
            break
            ;;
      esac
    done
  fi
fi

# ----------------- #
# RUN REFRAME TESTS #
# ----------------- #

# Settings file needed for Reframe
repo_root_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
rfm_settings_file=${repo_root_dir}/setup_files/settings.py

# Call Reframe based on passed arguments
if [[ "$mode" == "test file" ]]; then
  printf "Running tests from test file $val \n\n"
  reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests/$val -r --performance-report $reframe_opts
elif [[ "$mode" == "test category" ]]; then
  printf "Running tests from test category $val \n\n"
  reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -t $val -r --performance-report $reframe_opts
elif [[ "$mode" == "test name" ]]; then
  printf "Running test $val \n\n"
  reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -n $val -r --performance-report $reframe_opts
elif [[ "$mode" == "multiple tests" ]]; then
  printf "Running tests $val \n\n"
  reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -n "$val" -r --performance-report $reframe_opts
else
  printf "Running all tests\n\n"
  reframe -C ${rfm_settings_file} -c ${repo_root_dir}/tests -r --performance-report $reframe_opts
fi
