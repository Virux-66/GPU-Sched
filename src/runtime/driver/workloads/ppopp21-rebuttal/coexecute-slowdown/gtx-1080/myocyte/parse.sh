#!/bin/bash


echo "solver_2 kernel"
grep "solver_2" *.out | awk '{print $1" "$7}'
echo

echo "CUDA memcpy (H-to-D and D-to-H)"
grep "CUDA memcpy" *.out | awk '{print $1" "$5}'

