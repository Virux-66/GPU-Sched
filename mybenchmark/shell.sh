#!/bin/bash
set -x
PROFILER=/usr/local/cuda/nsight-compute-2021.1.1/ncu
BASE_PATH=/usr/local/cuda/samples
SECTION_FOLDER=/home/eailab/Tmp/Metrics/sections
SECTION=Arithmetic_Intensity
PROFILER_FLAG="-c 1 --section-folder ${SECTION_FOLDER} --section ${SECTION}"
RESULT=/home/eailab/Tmp/mybenchmark/profile.result

BENCHMARK=(
	0_Simple/matrixMul
	0_Simple/vectorAdd
	2_Graphics/simpleGL
	2_Graphics/simpleTexture3D
	4_Finance/binomialOptions
	4_Finance/BlackScholes
	5_Simulations/nbody
	6_Advanced/FDTD3d	
)

EXECUTABLE=(
	matrixMul
	vectorAdd
	simpleGL	
	simpleTexture3D
	binomialOptions	
	BlackScholes
	nbody
	FDTD3d
)

for index in  ${!BENCHMARK[@]}
do
	cd ${BASE_PATH}/${BENCHMARK[index]}
	#make
	if [ ${EXECUTABLE} = "nbody" ]
	then
		${PROFILER} ${PROFILER_FLAG} ${EXECUTABLE[index]} -benchmark -numbodies=100000 >> ${RESULT}
	else
		${PROFILER} ${PROFILER_FLAG} ${EXECUTABLE[index]} >> ${RESULT}
	fi
	#make clean
done

