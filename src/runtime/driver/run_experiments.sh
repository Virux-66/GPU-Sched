#!/bin/bash

#set -x

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

BASE_PATH=/home/eailab/Tmp
BEMPS_SCHED_PATH=${BASE_PATH}/sched-build/runtime/sched
WORKLOADER_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver
#WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/2023
#WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/introduction
#WORKLOADS_PATH=${BASE_PATH}/GPU-Sched/src/runtime/driver/workloads/2023
#RESULTS_PATH=results-introduction
#RESULTS_PATH=results-kernel-level-L1Cache

if [ $# -ne 2 ]
then 
    echo ""
    echo "Usage: "
    echo "$0 <workloads_path> <raw_results_path>"
    exit 1
fi
WORKLOADS_PATH=$1   #workloads/2023
RESULTS_PATH=$2     #raw_results
#list all workloads file
cd ${WORKLOADS_PATH}
WORKLOADS=(*)       #get all workload files
cd ../..

#WORKLOADS=(
#   3080Ti_32jobs_1.wl
#    3080Ti_32jobs_2.wl
#    3080Ti_32jobs_3.wl
#    3080Ti_32jobs_4.wl
#    3080Ti_32jobs_5.wl
#    3080Ti_32jobs_6.wl
#    3080Ti_32jobs_7.wl
#    3080Ti_32jobs_8.wl
#)

ZERO_ARGS_ARR=(
    #4 # <-- will throw up to 4 jobs onto GPU0 at once
    8 # <-- will throw up to 8 jobs onto GPU0 at once
    #16 # <-- will throw up to 16 jobs onto GPU0 at once
)
SINGLE_ASSIGNMENT_ARGS_ARR=(
    1
    #2 # <-- 2-GPU system
    #4 # <-- 4-GPU system
)
CG_ARGS_ARR=(
    #2 # <-- Don't use for 4-GPU system. Don't use for 2-GPU system unless sanity checking. This is equivalent to single-assignment.
    #3 # <-- Don't use for 4-GPU system
    #4 # <-- Don't use for 4-GPU system unless sanity checking. This is equivalent to single-assignment.
    #5
    6
    #7
    #8
    #9
    #10
    #11
    #12
    #24
    #32
    #64
    #96
    #128
)
MGB_ARGS_ARR=(
    #4
    #6
    #8
    #10 # <-- ultimately used for ppopp21 2xp100 results
    #12
    #14
    #16 # <-- ultimately used for ppopp21 4xv100 results
    #18
    #20
    #24
    #32
    #64
    #96
    #128
    #24.10 # num procs . max jobs waiting for GPU
    #48.10 # num procs . max jobs waiting for GPU
    16
)
AI_ARGS_ARR=(
    32      #This value should be as large as possible such that scheduler could have many enough kernels to schedule. The environment machine has 48 cores.
)

declare -A SCHED_ALG_TO_ARGS_ARR=(
    [zero]="ZERO_ARGS_ARR"
    [single-assignment]="SINGLE_ASSIGNMENT_ARGS_ARR"
    #[cg]="CG_ARGS_ARR"
    [mgb_basic]="MGB_ARGS_ARR"
    #[ai-heuristic]="AI_ARGS_ARR"
    [ai-mgb_basic]="AI_ARGS_ARR"
    #[mgb_simple_compute]="MGB_ARGS_ARR"
    #[mgb]="MGB_ARGS_ARR"
)


export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/eailab/Tmp/sched-build/runtime/bemps:/home/eailab/Tmp/libstatus

rm -f /dev/shm/bemps

mkdir -p ${RESULTS_PATH}


for WORKLOAD in ${WORKLOADS[@]}; do
    for SCHED_ALG in "${!SCHED_ALG_TO_ARGS_ARR[@]}"; do
        #rm -f /dev/shm/bemps
        #echo ${SCHED_ALG}
        ARGS_ARR_STR=${SCHED_ALG_TO_ARGS_ARR[$SCHED_ALG]}
        #echo $ARGS_ARR_STR
        eval ARGS_ARR=\${${ARGS_ARR_STR}[@]}                #just for simplicity
        for ARGS in ${ARGS_ARR[@]}; do
            #echo $ARGS
            WORKLOAD_NO_EXT=`basename $WORKLOAD .wl`
            #ARGS=${SCHED_ALG_TO_ARGS[$SCHED_ALG]}
            EXPERIMENT_BASENAME=${RESULTS_PATH}/${WORKLOAD_NO_EXT}.${SCHED_ALG}.${ARGS}

            # yet another hack: cg needs to know jobs-per-cpu. And we need to
            # be able to control it from this bash driver. So pass it along.
            SCHED_ARGS=""
            if [ "${SCHED_ALG}" == "cg" ]; then
                SCHED_ARGS=${ARGS}
            fi

            echo "Launching scheduler for ${EXPERIMENT_BASENAME}"
            ${BEMPS_SCHED_PATH}/bemps_sched ${SCHED_ALG} ${SCHED_ARGS} uni-gpu \
              &> ${EXPERIMENT_BASENAME}.sched-log &
            SCHED_PID=$!
            echo "Scheduler is running with pid ${SCHED_PID}"

            # FIXME Adding a hacky sleep. We have an unsafe assumption, though we
            # have yet to see a problem manifest: The scheduler needs to initialize
            # the shared memory (bemps_sched_init()) before benchmarks run and try
            # to open it (bemps_init()). When using mgbd (mgb with dynamic job
            # pressure), the workloader itself could also fail without a sufficient
            # delay here.
            sleep 1

            echo "Launching workoader for ${EXPERIMENT_BASENAME}"
            ${WORKLOADER_PATH}/workloader.py \
              ${WORKLOADS_PATH}/${WORKLOAD}  \
              ${SCHED_ALG} \
              ${ARGS} \
              &> ${EXPERIMENT_BASENAME}.workloader-log &            # > is truncate while >> is append. ${EXPERIMENT_BASENAME}.workloader-log
            WORKLOADER_PID=$!                                       # is created if it doesn't exist. 
            echo "Workloader is running with pid ${WORKLOADER_PID}"

            echo "Waiting for workloader to complete"
            wait ${WORKLOADER_PID}

            echo "Workloader done"
            echo "Killing scheduler"
            kill -2 ${SCHED_PID}
            sleep 5 # maybe a good idea before moving sched-stats.out
            mv ./sched-stats.out ${EXPERIMENT_BASENAME}.sched-stats
            mv ./sched_gpu.csv ${EXPERIMENT_BASENAME}-sched_gpu.csv
            mv ./sched_mem.csv ${EXPERIMENT_BASENAME}-sched_mem.csv

            echo "Completed experiment for ${EXPERIMENT_BASENAME}"


        done

    done
done

echo "All experiments complete"
echo "Exiting normally"
