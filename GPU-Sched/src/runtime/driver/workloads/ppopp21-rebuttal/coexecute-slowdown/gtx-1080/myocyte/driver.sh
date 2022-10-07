#!/bin/bash

# Note to future self: never got beyond 64 jobs because it was throwing errors
# already on sloppyjoe at that count
#for j in 1 2 4 8 16 32 64 128 256; do

#for j in 1 2 4 8 16 32 64; do
for j in 1 2 4 8 16; do
    echo "Running ${j}-job workload..."
    bash coexecute_slowdown_${j}jobs_1.wl &> result_${j}jobs.out
done
