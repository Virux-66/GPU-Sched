import random

TOTAL_NUM_BENCHMAKR='32'
BASE_PATH='/home/eailab/Tmp/GPU-Sched'
BENCHMARK_PATH=BASE_PATH + '/mybenchmark'
WORKLOAD_PATH=BASE_PATH + '/src/runtime/driver/workloads/2023'
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

compute_memory_ratio=[ 
    (1,1), (3,1), (5,3),
    (7,1), (1,3), (3,5),
    (1,7)
]

workloads_prefix=[
    0, 1, 2, 3, 4, 5, 6
]


for index in range(len(compute_memory_ratio)):
    ratio=compute_memory_ratio[index]
    ratio_compute=(int)(ratio[0])
    ratio_memory=(int)(ratio[1])
    num_compute=(int)(((int)(TOTAL_NUM_BENCHMAKR))*(ratio_compute/(ratio_compute+ratio_memory)))
    num_memory=(int)(((int)(TOTAL_NUM_BENCHMAKR))*(ratio_memory/(ratio_compute+ratio_memory)))
    #print(num_compute,num_memory)
    file_name=FILE_NAME_BASE+(str)(index)+'.wl' 
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