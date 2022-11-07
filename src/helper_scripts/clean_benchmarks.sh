#!/bin/bash


RODINIA_BMARK_PATHS=(
    Benchmarks/rodinia_cuda_3.1/cuda/backprop
    Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v1
    Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2
    Benchmarks/rodinia_cuda_3.1/cuda/lavaMD
    Benchmarks/rodinia_cuda_3.1/cuda/nw
    Benchmarks/rodinia_cuda_3.1/cuda/dwt2d
    Benchmarks/rodinia_cuda_3.1/cuda/bfs
)

# Clean Rodinia benchmarks
for BMARK_PATH in ${RODINIA_BMARK_PATHS[@]};do
    echo "Cleaning benchmark: "$BMARK_PATH
    cd $BMARK_PATH
    make clean
done

#Clean Darknet
cd Benchmarks/darknet
make clean