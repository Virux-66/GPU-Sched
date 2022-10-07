#!/bin/bash

# rough script that I may need to reuse.
# use case: You had to capture 2 separate nvprofile runs because trying to do
# it in one shot causes errors ("insufficient ...").
# This script assumes folder structure and other things, but the idea is to
# combine the "_2" and "_3" workloads into a "_1" workload file so that it
# can be parsed by the normal nvprof stuff we already have.

cat twos/basic/p100_16_32jobs_2.mgb_basic.16.workloader-log threes/basic/p100_16_32jobs_3.mgb_basic.16.workloader-log >> combined/p100_16_32jobs_1.mgb_basic.16.workloader-log

cat twos/basic/p100_25_32jobs_2.mgb_basic.16.workloader-log threes/basic/p100_25_32jobs_3.mgb_basic.16.workloader-log >> combined/p100_25_32jobs_1.mgb_basic.16.workloader-log

cat twos/basic/p100_33_32jobs_2.mgb_basic.16.workloader-log threes/basic/p100_33_32jobs_3.mgb_basic.16.workloader-log >> combined/p100_33_32jobs_1.mgb_basic.16.workloader-log

cat twos/basic/p100_50_32jobs_2.mgb_basic.16.workloader-log threes/basic/p100_50_32jobs_3.mgb_basic.16.workloader-log >> combined/p100_50_32jobs_1.mgb_basic.16.workloader-log



cat twos/reg/p100_16_32jobs_2.mgb.16.workloader-log threes/reg/p100_16_32jobs_3.mgb.16.workloader-log >> combined/p100_16_32jobs_1.mgb.16.workloader-log

cat twos/reg/p100_25_32jobs_2.mgb.16.workloader-log threes/reg/p100_25_32jobs_3.mgb.16.workloader-log >> combined/p100_25_32jobs_1.mgb.16.workloader-log

cat twos/reg/p100_33_32jobs_2.mgb.16.workloader-log threes/reg/p100_33_32jobs_3.mgb.16.workloader-log >> combined/p100_33_32jobs_1.mgb.16.workloader-log

cat twos/reg/p100_50_32jobs_2.mgb.16.workloader-log threes/reg/p100_50_32jobs_3.mgb.16.workloader-log >> combined/p100_50_32jobs_1.mgb.16.workloader-log
