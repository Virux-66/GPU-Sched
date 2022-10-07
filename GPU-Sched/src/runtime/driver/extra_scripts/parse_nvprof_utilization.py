#!/usr/bin/env python3
import sys
import re

#
# We're parsing from the workloader logs the nvprof data, which looks like this:
#
#   ==10222== NVPROF is profiling process 10222, command: /home/cc/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 8192 8192 0 127 0 127 0.5 2
#   ==10222== Profiling application: /home/cc/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda/srad/srad_v2/srad 8192 8192 0 127 0 127 0.5 2
#   ==10222== Profiling result:
#   Type  Time(%)      Time     Calls       Avg       Min       Max  Name
#   GPU activities:   44.22%  61.423ms         2  30.712ms  30.616ms  30.807ms  [CUDA memcpy HtoD]
#   43.04%  59.784ms         2  29.892ms  29.863ms  29.921ms  [CUDA memcpy DtoH]
#   6.95%  9.6529ms         2  4.8265ms  4.8244ms  4.8285ms  srad_cuda_1(float*, float*, float*, float*, float*, float*, int, int, float)
#   5.78%  8.0336ms         2  4.0168ms  4.0153ms  4.0183ms  srad_cuda_2(float*, float*, float*, float*, float*, float*, int, int, float, float)
#   API calls:   58.53%  238.20ms         6  39.700ms  420.07us  236.06ms  cudaMalloc
#   34.27%  139.48ms         4  34.869ms  30.676ms  38.940ms  cudaMemcpy
#   6.09%  24.791ms         6  4.1318ms  627.35us  8.0417ms  cudaFree
#   0.57%  2.3255ms         2  1.1627ms  1.1483ms  1.1772ms  cuDeviceTotalMem
#   0.44%  1.8065ms       194  9.3110us     350ns  375.06us  cuDeviceGetAttribute
#   0.04%  171.61us         4  42.901us  9.3680us  100.33us  cudaLaunchKernel
#   0.04%  165.15us         2  82.573us  79.480us  85.667us  cuDeviceGetName
#   0.00%  12.487us         1  12.487us  12.487us  12.487us  cudaSetDevice
#   0.00%  9.7700us         1  9.7700us  9.7700us  9.7700us  cudaThreadSynchronize
#   0.00%  7.3340us         2  3.6670us  3.5260us  3.8080us  cuDeviceGetPCIBusId
#   0.00%  5.7000us         4  1.4250us     460ns  2.5100us  cuDeviceGet
#   0.00%  2.6640us         3     888ns     353ns  1.2200us  cuDeviceGetCount
#   0.00%  1.2390us         2     619ns     543ns     696ns  cuDeviceGetUuid
#
# 
# We're also parsing the beacon info (to grab mem_B and thread_blocks), which
# looks like this:
# 
#   2166229609221 bemps_beacon: pid 21163 , bemps_tid 0 , mem_B 3872176000 , warps 3781264 , thread_blocks 236329
#
#
# Lastly, we have to grab the total benchmark time, which looks like this
#
#   Worker 5: TOTAL_BENCHMARK_TIME 124 93.53748178482056                            
#
#
# The end goal is to be able to spit out utilizaiton of memory and thread
# blocks across the GPUs.
#


pid_to_kernel_details = {}
pid_to_kernel_time    = {}
total_experiment_time_ms = None


def convert_time_to_ms(t):
    #print(t)
    m = re.match(r"([0-9]*\.*[0-9]*)([a-z]*)", t)
    if m:
        v = float(m.group(1))
        if   'ns' == m.group(2):
            v /= 1000000
        elif 'us' == m.group(2):
            v /= 1000
        elif 'ms' == m.group(2):
            pass
        elif 's' == m.group(2):
            v *= 1000
        else:
            print('Unexpected time: {}'.format(t))
            sys.exit(1)
    else:
        print('Unexpected time: {}'.format(t))
        sys.exit(1)
    #print(v)
    return v


def parse(workload_log):
    global pid_to_kernel_details
    global pid_to_kernel_time
    global total_experiment_time_ms

    count = 0
    with open(workload_log) as f:
        inside_1 = False
        inside_2 = False
        pid = None
        for line in f:
            line = line.strip()

            if "bemps_beacon" in line:
                line = line.split()
                pid = line[3]
                mem_B = int(line[9])
                thread_blocks = int(line[15])
                pid_to_kernel_details[pid] = (mem_B, thread_blocks)
            
            # case: line begins a section of nvprof output
            elif "NVPROF" in line:
                assert inside_1 == False
                assert inside_2 == False
                inside_1 = True

            # case: we're inside a section of nvprof output.
            elif inside_1:
                assert inside_2 == False
                # case: we are starting to see the lines that we're interested
                # in. They start once we see 'GPU activities'.
                if 'GPU activities' in line:
                    line = line.split()
                    t = convert_time_to_ms(line[3])
                    pid_to_kernel_time[pid] = t
                    inside_1 = False
                    inside_2 = True
            elif inside_2:
                assert inside_1 == False
                # case: we were seeing lines we cared about, they end once we
                # see 'API calls'.
                if 'API calls' in line:
                    inside_2 = False
                # case: we are still seeing lines we care about
                else:
                    line = line.split()
                    t = convert_time_to_ms(line[1])
                    pid_to_kernel_time[pid] += t

            elif 'TOTAL_EXPERIMENT_TIME' in line:
                total_experiment_time_ms = float(line.split()[3]) * 1000

        assert inside_1 == False
        assert inside_2 == False
    #print('sanity check, count of items added should equal num-jobs: {}'.format(count))



def usage_and_exit():
    print('Usage: TODO. check script. probably need to pass sa and mgb workloader logs')
    sys.exit(1)



if len(sys.argv) != 3:
    usage_and_exit()
sa_workloader_log  = sys.argv[1] 
mgb_workloader_log = sys.argv[2] 


#parse(sa_workloader_log)
parse(mgb_workloader_log)

print(total_experiment_time_ms)
#for pid, kernel_details in pid_to_kernel_details.items():

running_mem = 0
running_tb  = 0
#for pid in pid_to_kernel_details:
for pid in pid_to_kernel_time:

    mem_B, thread_blocks = pid_to_kernel_details[pid]
    
    # 'kernel_time' correctly includes times for more than 1 kernel for a given
    # pid, if there was one
    kernel_time = pid_to_kernel_time[pid] 

    #print('{} {} {} {}'.format(pid, mem_B, thread_blocks, kernel_time))
    #print('{}'.format(kernel_time / total_experiment_time_ms))
    #foo = kernel_time / total_experiment_time_ms
    #print('{}'.format(type( mem_B)))
    #print('{}'.format(foo * mem_B))

    running_mem += kernel_time / total_experiment_time_ms * mem_B
    running_tb  += kernel_time / total_experiment_time_ms * thread_blocks

# v100, assuming we pad for safety
TOTAL_GPU_MEM = 10000 * 1024 * 1024 * 4
TOTAL_GPU_TB  = 80 * 32 * 4

print('gpu memory utilization:       {}'.format(running_mem / TOTAL_GPU_MEM))
print('gpu thread block utilization: {}'.format(running_tb  / TOTAL_GPU_TB))
print('FIXME: pid-to-kernel-details and pid-to-kernel-time should have the same number of elements (i.e. pids), but they dont at the moment. this means we are missing log data and so the result is innaccurate')
