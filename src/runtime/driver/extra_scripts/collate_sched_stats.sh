#!/bin/bash

#BASE_FLD=/home/cc/GPU-Sched/src/runtime/driver
BASE_FLD=/home/ubuntu/GPU-Sched/src/runtime/driver
SUFFIX=sched-stats


# 16 jobs
#RESULTS_FLD=results-2020.08.13/16jobs
#FILES=(
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_16jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_16jobs_0.cg.12.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_16jobs_0.mgb.16.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_16jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_16jobs_0.cg.6.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_16jobs_0.mgb.16.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_16jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_16jobs_0.cg.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_16jobs_0.mgb.16.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_16jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_16jobs_0.cg.12.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_16jobs_0.mgb.16.${SUFFIX}
#)
# 32 jobs
RESULTS_FLD=results-2020.08.13/32jobs
FILES=(
    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.single-assignment.4.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.cg.6.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.mgb.16.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.single-assignment.4.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.cg.4.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.mgb.16.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.single-assignment.4.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.cg.6.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.mgb.16.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.single-assignment.4.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.cg.6.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.mgb.16.${SUFFIX}
)



echo "sched-stats"
for FILE in ${FILES[@]}; do
    echo `basename ${FILE}`
    cat ${FILE}
    echo
done
