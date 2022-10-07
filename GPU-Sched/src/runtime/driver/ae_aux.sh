#!/bin/bash

function ae_run() {

    mkdir -p results

    for WORKLOAD in ${WORKLOADS[@]}; do
        for SCHED_ALG in "${!SCHED_ALG_TO_ARGS_ARR[@]}"; do

            #echo ${SCHED_ALG}
            ARGS_ARR_STR=${SCHED_ALG_TO_ARGS_ARR[$SCHED_ALG]}
            #echo $ARGS_ARR_STR
            eval ARGS_ARR=\${${ARGS_ARR_STR}[@]}
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
                ${BEMPS_SCHED_PATH}/bemps_sched ${SCHED_ALG} ${SCHED_ARGS} \
                  &> ${EXPERIMENT_BASENAME}.sched-log &
                SCHED_PID=$!
                echo "Scheduler is running with pid ${SCHED_PID}"

                # FIXME Adding a hacky sleep. We have an unsafe assumption, though we
                # have yet to see a problem manifest: The scheduler needs to initialize
                # the shared memory (bemps_sched_init()) before benchmarks run and try
                # to open it (bemps_init()). When using mgbd (mgb with dynamic job
                # pressure), the workloader itself could also fail without a sufficient
                # delay here.
                sleep 3

                echo "Launching workoader for ${EXPERIMENT_BASENAME}"
                ${WORKLOADER_PATH}/workloader.py \
                  ${WORKLOADS_PATH}/${WORKLOAD}  \
                  ${SCHED_ALG} \
                  ${ARGS} \
                  &> ${EXPERIMENT_BASENAME}.workloader-log &
                WORKLOADER_PID=$!
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
}
