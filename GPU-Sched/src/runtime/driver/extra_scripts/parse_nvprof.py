#!/usr/bin/env python3
import sys
import re
import statistics
import math
from pprint import pprint


def geomean(xs):
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))



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



# Basic, basic parse. If you pass a workloader log as an argument to this
# script this will dump only the nvprof lines.
def dump_nvprof_lines_from_log():
    with open(sys.argv[1]) as f:
        inside = False
        for line in f:
            if "NVPROF" in line:
                inside = True
            if inside:
                print(line.strip())
            if '' == line.strip():
                inside = False








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






def parse(workload_log, cmd_to_invocations):
    count = 0
    with open(workload_log) as f:
        inside_1 = False
        inside_2 = False
        cmd_str = ''
        for line in f:
            line = line.strip()
            
            # case: line begins a section of nvprof output
            if "NVPROF" in line:
                COMMAND_NEEDLE = ', command: '
                cmd_str = line[line.find(COMMAND_NEEDLE) + len(COMMAND_NEEDLE):]
                if cmd_str not in cmd_to_invocations:
                    cmd_to_invocations[cmd_str] = []
                invocation = {}
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
                    kernel_name = ' '.join(line[8:])
                    invocation[kernel_name] = t
                    inside_1 = False
                    inside_2 = True
            elif inside_2:
                assert inside_1 == False
                # case: we were seeing lines we cared about, they end once we
                # see 'API calls'.
                if 'API calls' in line:
                    inside_2 = False
                    cmd_to_invocations[cmd_str].append(invocation)
                    cmd_str = ''
                    count += 1
                # case: we are still seeing lines we care about
                else:
                    line = line.split()
                    t = convert_time_to_ms(line[1])
                    kernel_name = ' '.join(line[6:])
                    invocation[kernel_name] = t
                    

        assert inside_1 == False
        assert inside_2 == False
    #print('sanity check, count of items added should equal num-jobs: {}'.format(count))

def calc_average_kernel_times(cmd_to_invocations, avgs, which):
    for cmd, invocations in cmd_to_invocations.items():
        for invocation in invocations:
            for kernel_name, t in invocation.items():
                k = cmd + kernel_name 
                if k not in avgs[which]:
                    avgs[which][k] = 0
                avgs[which][k] += t
        avgs[which][k] = 1.0 * avgs[which][k] / len(invocations)


def report():
    # two-element array where
    #   avgs[SA_AVGS][some_key] = the average time for that specific command and kernel
    # 'some_key' is a concatenation of the benchmark cmd + the kernel name (which
    # will be unique) 
    SA_AVGS  = 0
    MGB_AVGS = 1
    avgs = [{}, {}]

    calc_average_kernel_times(cmd_to_invocations_sa, avgs, SA_AVGS)
    calc_average_kernel_times(cmd_to_invocations_mgb, avgs, MGB_AVGS)

    slowdowns_data_movement = []
    slowdowns_kernels       = []

    for k in avgs[SA_AVGS]:
        mgb_avg_kernel_time = avgs[MGB_AVGS][k]
        sa_avg_kernel_time  = avgs[SA_AVGS][k]
        #print(k)
        #print(mgb_avg_kernel_time)
        #print(sa_avg_kernel_time)
        slowdown = 1.0 * mgb_avg_kernel_time / sa_avg_kernel_time
        if '[CUDA memcpy DtoH]' in k or '[CUDA memcpy HtoD]' in k:
            #print(slowdown)
            slowdowns_data_movement.append(slowdown)
        else:
            slowdowns_kernels.append(slowdown)
        #print('slowdown: {}'.format(slowdown))
        #print()

    avg_kernel_slowdown            = statistics.mean(slowdowns_kernels)
    avg_data_movement_slowdown     = statistics.mean(slowdowns_data_movement)
    geomean_kernel_slowdown        = geomean(slowdowns_kernels)
    geomean_data_movement_slowdown = geomean(slowdowns_data_movement)
    print('mean_kernel_slowdown {}'.format(avg_kernel_slowdown))
    print('geomean_kernel_slowdown {}'.format(geomean_kernel_slowdown))
    print('mean_data_movement_slowdown {}'.format(avg_data_movement_slowdown))
    print('geomean_data_movement_slowdown {}'.format(geomean_data_movement_slowdown))
 

    




def usage_and_exit():
    print('Usage: TODO. check script. probably need to pass sa and mgb workloader logs')
    sys.exit(1)


#dump_nvprof_lines_from_log()
#sys.exit(1)





# These cmd-to-invocation dictionaries look like this:
# {
#     full-cmd-string: [
#     # index 0 is the first time this command executed in the workload
#     {
#         full-kernel-name-1: time,
#         full-kernel-name-2: time
#     },
#     # index 1 is just another instance of this command executing in th workload
#     {
#         full-kernel-name-1: time,
#         full-kernel-name-2: time,
#     },
#     ]
# }
# the full-cmd-string is some command, e.g "./backprop 100"
# Every time it got executed in the workload, it'll have an entry in its
# array. And for each of those entries, it's another map of the name of that
# kernl e.g. "srad(float *, float*)", and the total time that that kernel
# spent executing on the GPU.
cmd_to_invocations_sa  = {}
cmd_to_invocations_mgb = {}



if len(sys.argv) != 3:
    usage_and_exit()
sa_workloader_log  = sys.argv[1] 
mgb_workloader_log = sys.argv[2] 


parse(sa_workloader_log, cmd_to_invocations_sa)
parse(mgb_workloader_log, cmd_to_invocations_mgb)


#print('SINGLE ASSIGNMENT')
#pprint(cmd_to_invocations_sa)
#print('MGB')
#pprint(cmd_to_invocations_mgb)
for k in cmd_to_invocations_sa:
    assert k in cmd_to_invocations_mgb
for k in cmd_to_invocations_mgb:
    assert k in cmd_to_invocations_sa


report()
