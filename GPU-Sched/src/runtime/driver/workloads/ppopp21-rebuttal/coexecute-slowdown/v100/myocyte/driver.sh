#!/bin/bash


#for j in 1 2 4 8 16 32 64; do
#for j in 128 256 512; do
for j in 1 2 4 8 16 32 48; do # only up to 48 procs on volta mps
    echo "Running ${j}-job workload..."
    bash coexecute_slowdown_${j}jobs_1.wl &> result_${j}jobs.out
done
