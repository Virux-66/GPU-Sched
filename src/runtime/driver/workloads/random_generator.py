import random

TOTAL_NUM_BENCHMAKR='32'
BASE_PATH='/home/eailab/Tmp/GPU-Sched'
BENCHMARK_PATH=BASE_PATH + '/mybenchmark'
#WORKLOAD_PATH=BASE_PATH + '/src/runtime/driver/workloads/2023'
WORKLOAD_PATH=BASE_PATH + '/src/runtime/driver/workloads/introduction'
GPU='3080Ti'
FILE_NAME_BASE=WORKLOAD_PATH + '/' + GPU + '_'+TOTAL_NUM_BENCHMAKR +'jobs_'

compute_bound_jobs = [
    BENCHMARK_PATH+'/binomialOptions_64/binomialOptions_64',
    BENCHMARK_PATH+'/binomialOptions_1024/binomialOptions_1024',
    BENCHMARK_PATH+'/matrixMul_1024/matrixMul_1024'
]
memory_bound_jobs = [
    BENCHMARK_PATH+'/BlackScholes/BlackScholes',
    BENCHMARK_PATH+'/histogram/histogram',
    BENCHMARK_PATH+'/matrixMul_2048/matrixMul_2048',
    BENCHMARK_PATH+'/scan/scan',
    BENCHMARK_PATH+'/vectorAdd/vectorAdd'
]

#This ratio is used to produce workloads whose comparison results are in evaluation part
compute_memory_ratio=[ 
    (1,0), (1,1), (3,1), 
    (5,3), (7,1), (1,3), 
    (3,5), (1,7)
]
#This ratio is used to produce workloads in intruction part whose comparison results are in introduction part

compute_memory_ratio_introduction=[
    (1,0), (0,1), (1,3)
]

ratio_used=compute_memory_ratio_introduction

workloads_prefix=[
    1, 2, 3, 4, 5, 6, 7, 8
]


for index in range(len(ratio_used)):
    ratio=ratio_used[index]
    ratio_compute=(int)(ratio[0])
    ratio_memory=(int)(ratio[1])
    num_compute=(int)(((int)(TOTAL_NUM_BENCHMAKR))*(ratio_compute/(ratio_compute+ratio_memory)))
    num_memory=(int)(((int)(TOTAL_NUM_BENCHMAKR))*(ratio_memory/(ratio_compute+ratio_memory)))
    #print(num_compute,num_memory)
    file_name=FILE_NAME_BASE+(str)(index+1)+'.wl' 
    #print(file_name)

    benchmarks=[]
    for compute_index in range(num_compute):
        random_number = random.randint(1,100)
        compute_benchmark_index=random_number % len(compute_bound_jobs)
        benchmarks.append(compute_bound_jobs[compute_benchmark_index])

    for memory_index in range(num_memory):
        random_number = random.randint(1,100)
        memory_benchmark_index = random_number % len(memory_bound_jobs) 
        benchmarks.append(memory_bound_jobs[memory_benchmark_index])

    random.shuffle(benchmarks) 

    with open(file_name,'w+') as f:
       for ben_index in range(len(benchmarks)):
            if ben_index < len(benchmarks)-1:
                f.write(benchmarks[ben_index]+'\n')
            else:
                f.write(benchmarks[ben_index])