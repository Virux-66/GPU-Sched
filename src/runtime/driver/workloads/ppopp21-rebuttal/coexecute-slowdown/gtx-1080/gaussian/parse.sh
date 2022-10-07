#!/bin/bash


echo "Fan2 kernel"
grep Fan2 *.out | awk '{print $1" "$7}'
echo

echo "Fan1 kernel"
grep Fan1 *.out | awk '{print $1" "$5}'
echo

echo "CUDA memcpy (H-to-D and D-to-H)"
grep "CUDA memcpy" *.out | awk '{print $1" "$5}'

