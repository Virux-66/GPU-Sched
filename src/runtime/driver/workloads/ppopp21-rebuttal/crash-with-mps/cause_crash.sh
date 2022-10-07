#!/bin/bash

echo "Kicking off jobs"

(time /home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/dwt2d/dwt2d /home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data/dwt2d/rgb.bmp -d 16384x16384 -f -5 -l 3; echo "returned: $?") &
(time /home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/dwt2d/dwt2d /home/ubuntu/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data/dwt2d/rgb.bmp -d 16384x16384 -f -5 -l 3; echo "returned: $?") &

echo "Waiting for jobs to complete..."
wait
