#!/bin/bash


RODINIA_BMARK_PATHS=(
    /home/eailab/Tmp/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/backprop
    /home/eailab/Tmp/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1
    /home/eailab/Tmp/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2
    /home/eailab/Tmp/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/lavaMD
    /home/eailab/Tmp/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/nw
    #Benchmarks/rodinia_cuda_3.1/cuda/dwt2d
    /home/eailab/Tmp/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/bfs
)

# Clean Rodinia benchmarks
for BMARK_PATH in ${RODINIA_BMARK_PATHS[@]};do
    echo "Cleaning benchmark: "$BMARK_PATH
    cd $BMARK_PATH
    make clean
done