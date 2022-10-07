#!/bin/bash

#set -x

#
# Structure of this bash script:
#
#   for each scheduling algorithm
#     for each workload
#       Start the scheduler
#       Start the workload driver
#       (Workload driver completes)
#       Stop scheduler
#       Parse workload driver output
#       Move scheduler results to results folder
#

source ae_aux.sh

BASE_PATH=/home/ubuntu
BEMPS_SCHED_PATH=${BASE_PATH}/GPU-Sched/build/runtime/sched
WORKLOADER_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver
WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/ppopp22-ae
RESULTS_PATH=results


function usage_and_exit() {
    echo
    echo "  Usage:"
    echo "    $0 <figure>"
    echo
    echo "    Figure must be one of:"
    echo "      {figure-4, figure-5, figure-7}"
    echo
    exit 1
}




if [ "$1" == "figure-4" ]; then
    echo "Running experiments for Figure 4 results"
    WORKLOADS=(
        v100_16_16jobs_0.wl
        v100_25_16jobs_0.wl
        v100_33_16jobs_0.wl
        v100_50_16jobs_0.wl
        v100_16_32jobs_0.wl
        v100_25_32jobs_0.wl
        v100_33_32jobs_0.wl
        v100_50_32jobs_0.wl
    )
    MGB_ARGS_ARR=(
        16
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [mgb_basic]="MGB_ARGS_ARR"
        [mgb]="MGB_ARGS_ARR"
    )
elif [ "$1" == "figure-5" ]; then
    echo "Running experiments for Figure 5 results"
    WORKLOADS=(
        v100_16_16jobs_0.wl
        v100_25_16jobs_0.wl
        v100_33_16jobs_0.wl
        v100_50_16jobs_0.wl
        v100_16_32jobs_0.wl
        v100_25_32jobs_0.wl
        v100_33_32jobs_0.wl
        v100_50_32jobs_0.wl
    )
    SINGLE_ASSIGNMENT_ARGS_ARR=(
        4
    )
    CG_ARGS_ARR=(
        6
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [single-assignment]="SINGLE_ASSIGNMENT_ARGS_ARR"
        [cg]="CG_ARGS_ARR"
    )
elif [ "$1" == "figure-7" ]; then
    echo "Running experiments for Figure 7 results"
    WORKLOADS=(
        v100_2_8jobs_0.wl
        v100_0_8jobs_1.wl
        v100_0_8jobs_2.wl
        v100_0_8jobs_3.wl
    )
    ZERO_ARGS_ARR=(
        8
    )
    MGB_ARGS_ARR=(
        8
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [zero]="ZERO_ARGS_ARR"
        [mgb_basic]="MGB_ARGS_ARR"
    )
else
    usage_and_exit
fi




ae_run



echo "Experiments complete"
echo "Exiting normally"
