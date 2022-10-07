#!/bin/bash


#BASE_FLD=/home/cc/GPU-Sched/src/runtime/driver
#BASE_FLD=/home/ubuntu/GPU-Sched/src/runtime/driver
BASE_FLD=/home/rudy/wo/gpu/GPU-Sched/src/runtime/driver
SUFFIX=workloader-log

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
#RESULTS_FLD=results-2020.08.13/32jobs
#FILES=(
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.cg.6.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.mgb.16.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.cg.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.mgb.16.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.cg.6.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.mgb.16.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.single-assignment.4.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.cg.6.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.mgb.16.${SUFFIX}
#)
#RESULTS_FLD=results/mgb-regular
#FILES=(
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_50_32jobs_0.mgb.10.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_33_32jobs_0.mgb.10.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_25_32jobs_0.mgb.10.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.single-assignment.2.${SUFFIX}
#    ${BASE_FLD}/${RESULTS_FLD}/p100_16_32jobs_0.mgb.10.${SUFFIX}
#)
RESULTS_FLD=results/hand-picked/regular
FILES=(
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_8jobs_1.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_8jobs_1.mgb_basic.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_8jobs_1.mgb.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_16jobs_1.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_16jobs_1.mgb_basic.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_16jobs_1.mgb.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_32jobs_1.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_32jobs_1.mgb_basic.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_32jobs_1.mgb.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_64jobs_1.single-assignment.2.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_64jobs_1.mgb_basic.10.${SUFFIX}
    ${BASE_FLD}/${RESULTS_FLD}/hand_picked_64jobs_1.mgb.10.${SUFFIX}
)


echo "TIME_SINCE_START"
echo "Sorted-based-on-job-index"
for FILE in ${FILES[@]}; do
    echo `basename ${FILE}`
    grep "TIME_SINCE_START" ${FILE} | awk '{print $4" "$5}' | sort -n
done

#echo
#echo
#echo
#echo
#
#echo "Sorted-based-on-time-since-start"
#for FILE in ${FILES[@]}; do
#    echo `basename ${FILE}`
#    grep "TIME_SINCE_START" ${FILE} | awk '{print $5" "$4}' | sort -n | awk '{print $2" "$1}'
#done
