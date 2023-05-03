#!/bin/bash

# Get hosts to see what state they are in (this returns the host name of every process, not unique list of hosts)
hostnames=($(srun hostname))
# Create logging directory which stores the dmesg output log files
logdir=logs/${SLURM_JOB_ID}
if [[ ! -d $logdir ]]
then
    mkdir -p $logdir
fi
# Check node health via `dmesg` and node memory via `free`
hostnames=($(for hn in "${hostnames[@]}"; do echo "${hn}"; done | sort -u)) # Get unique nids
timestamp=$(date -Iseconds)
for h in ${hostnames[@]}
do
    srun -w ${h} --nodes=1 --ntasks=1 --ntasks-per-node=1 --mem=1GB dmesg -T > ${logdir}/node-state.${h}.${timestamp}.job-${SLURM_JOB_ID}.txt
    srun -w ${h} --nodes=1 --ntasks=1 --ntasks-per-node=1 --mem=1GB free -h >> ${logdir}/node-state.${h}.${timestamp}.job-${SLURM_JOB_ID}.txt
    numerr=$(grep -aic "error" ${logdir}/node-state.${h}.${timestamp}.job-${SLURM_JOB_ID}.txt)
    echo "There are $numerr errors in the dmesg output for node ${h}"
done
