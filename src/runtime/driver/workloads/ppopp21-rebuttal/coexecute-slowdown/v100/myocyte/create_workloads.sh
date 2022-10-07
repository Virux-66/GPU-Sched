#!/bin/bash

CMD="nvprof /home/rudy/wo/gpu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/myocyte/myocyte.out 1000 100 1 &"


# Note to future self: never got beyond 64 jobs because it was throwing errors
# already on sloppyjoe at that count
#for j in 8 16 32 64 128 256; do

for j in 8 16 32 64; do
    OUT_FILE="coexecute_slowdown_${j}jobs_1.wl"
    for i in $(seq 1 $j); do
        echo $CMD >> $OUT_FILE
    done
    echo "wait" >> $OUT_FILE
done
