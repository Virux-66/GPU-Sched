#!/usr/bin/env python3

import sys
import random
import time
import copy

#BASE_PATH = '/home/rudy/wo/gpu'
BASE_PATH = '/home/cc'
#BASE_PATH = '/home/ubuntu'
RODINIA_BMARK_PATH = BASE_PATH+'/GPU-Sched/Benchmarks/rodinia_cuda_3.1/cuda'
RODINIA_DATA_PATH  = BASE_PATH+'/GPU-Sched/Benchmarks/rodinia_cuda_3.1/data'


GPU_TO_MEM = {
    'gtx1080':  8 * 1024 * 1024 * 1024, #  8 GB
    'k80':     12 * 1024 * 1024 * 1024, # 12 GB
    'p100':    16 * 1024 * 1024 * 1024, # 16 GB
    'v100':    16 * 1024 * 1024 * 1024, # 16 GB
}


# all_jobs is a list of tuples:
#   (max kernel memory size, benchmark command that produces it)
all_jobs = [
    # ~ 10 MB
    (   10074152, 'particlefilter/particlefilter_float -x 256 -y 256 -z 100 -np 10000'),
    (   20477952, 'cfd/euler3d {}/cfd/missile.domn.0.2M'.format(RODINIA_DATA_PATH)),
    (   20523432, 'b+tree/b+tree.out file {}/b+tree/mil.txt ' \
                  'command {}/b+tree/command.txt'.format(RODINIA_DATA_PATH,
                                                         RODINIA_DATA_PATH)),
    (   32505856, 'dwt2d/dwt2d {}/dwt2d/rgb.bmp -d 1024x1024 -f -5 -l 3'.format(RODINIA_DATA_PATH)),
    (   33562624, 'gaussian/gaussian -s 2048'),
    (   33587208, 'nw/needle 2048 10'),
    (   37464000, 'myocyte/myocyte.out 1000 100 1'),
    (   38999881, 'bfs/bfs {}/bfs/graph1MW_6.txt'.format(RODINIA_DATA_PATH)),
    (   40400000, 'pathfinder/pathfinder 100000 100 20'),
    (   43350996, 'heartwall/heartwall {}/heartwall/test.avi 20'.format(RODINIA_DATA_PATH)),
    (   50331648, 'hotspot/hotspot 2048 2 2 {}/hotspot/gt_temp_2048 ' \
                  '{}/hotspot/gt_power_2048 ' \
                  '{}/hotspot/output.out'.format(RODINIA_DATA_PATH,
                                                 RODINIA_DATA_PATH,
                                                 RODINIA_DATA_PATH)),
    (   61414952, 'particlefilter/particlefilter_float -x 512 -y 512 -z 100 -np 100000'),
    # ~ 100 MB
    (  100663296, 'srad/srad_v2/srad 2048 2048 0 127 0 127 0.5 2'),
    (  104893352, 'particlefilter/particlefilter_float -x 1024 -y 1024 -z 100 -np 100'),
    (  130023424, 'dwt2d/dwt2d {}/dwt2d/rgb.bmp -d 2048x2048 -f -5 -l 3'.format(RODINIA_DATA_PATH)),
    (  134234112, 'gaussian/gaussian -s 4096'),
    (  134283272, 'nw/needle 4096 10'),
    (  327148809, 'bfs/bfs {}/bfs/inputGen/graph8M.txt'.format(RODINIA_DATA_PATH)),
    (  374640000, 'myocyte/myocyte.out 1000 1000 1'),
    (  400400000, 'pathfinder/pathfinder 100000 1000 20'),
    (  402653184, 'srad/srad_v2/srad 4096 4096 0 127 0 127 0.5 2'),
    (  520093696, 'dwt2d/dwt2d {}/dwt2d/rgb.bmp -d 4096x4096 -f -5 -l 3'.format(RODINIA_DATA_PATH)),
    (  537001992, 'nw/needle 8192 10'),
    (  589299988, 'backprop/backprop 4194304'),
    # ~ 1 GB
    ( 1176502548, 'backprop/backprop 8388608'),
    ( 1308592369, 'bfs/bfs {}/bfs/inputGen/graph32M.txt'.format(RODINIA_DATA_PATH)),
    ( 1610612736, 'srad/srad_v2/srad 8192 8192 0 127 0 127 0.5 2'),
    ( 2080374784, 'dwt2d/dwt2d {}/dwt2d/rgb.bmp -d 8192x8192 -f -5 -l 3'.format(RODINIA_DATA_PATH)),
    ( 2147745800, 'nw/needle 16384 10'),
    ( 2350907668, 'backprop/backprop 16777216'),
    ( 3872176000, 'srad/srad_v1/srad 100 0.5 11000 11000'),
    # ~ 4 GB
    ( 4699717908, 'backprop/backprop 33554432'),
    ( 6442450944, 'srad/srad_v2/srad 16384 16384 0 127 0 127 0.5 2'),
    ( 7200240000, 'srad/srad_v1/srad 100 0.5 15000 15000'),
    ( 7856000000, 'lavaMD/lavaMD -boxes1d 100'),
    ( 8321499136, 'dwt2d/dwt2d {}/dwt2d/rgb.bmp -d 16384x16384 -f -5 -l 3'.format(RODINIA_DATA_PATH)),
    ( 8590458888, 'nw/needle 32768 10'),
    ( 9397338388, 'backprop/backprop 67108864'),
	# ~ 10 GB
    (10456336000, 'lavaMD/lavaMD -boxes1d 110'),
    (12800320000, 'srad/srad_v1/srad 100 0.5 20000 20000'),
    (13575168000, 'lavaMD/lavaMD -boxes1d 120'),


    #
    # Don't use:
    #

    # 50s to complete, when its memory footprint peers are around 0.5s - 15s
    #(  419466152, 'particlefilter/particlefilter_float -x 2048 -y 2048 -z 100 -np 100'),

    # 90s to complete, when its peers are around 5 - 30s
    #( 3686640000, 'myocyte/myocyte.out 10000 1000 1'),

    # similar... these are too long compared to peers
    #(   48000000, 'particlefilter/particlefilter_naive ' \
    #              '-x 128 -y 128 -z 10 -np 1000000'),
    #(  210834320, 'b+tree/b+tree.out file {}/b+tree/mil_gt.txt command ' \
    #              '{}/b+tree/command_gt.txt'.format(RODINIA_DATA_PATH,
    #                                                RODINIA_DATA_PATH)),
    #(  201326592, 'hotspot3D/3D 512 64 1000 {}/hotspot3D/power_512x64 ' \
    #              '{}/hotspot3D/temp_512x64 ' \
    #              '{}/hotspot3D/output.out'.format(RODINIA_DATA_PATH,
    #                                               RODINIA_DATA_PATH,
    #                                               RODINIA_DATA_PATH)),
    #(  654310889, 'bfs/bfs {}/bfs/inputGen/graph16M.txt'.format(RODINIA_DATA_PATH)),

    # seems halved to 12GB
    #(25769803776, 'srad/srad_v2/srad 32768 32768 0 127 0 127 0.5 2'),
]
small_jobs = []
medium_jobs = []
large_jobs = []
under_one = []   # jobs with < 1 GB of footprint
one_to_four = [] # jobs with 1 - 4 GB of footprint
over_four = []   # jobs with > 4 GB of footprint


# helper for "small", "medium", and "large" mixes
job_mix_to_jobs = {
    'small': small_jobs,
    'medium': medium_jobs,
    'large': large_jobs,
}

# helper for "random" mixes
job_bufs = [
    small_jobs,
    medium_jobs,
    large_jobs
]

# helper for "50", "33", etc. mixes
job_mix_to_denominator = {
    '50': 2,
    '33': 3,
    '25': 4,
    '16': 6,
    'v2-90': 10
}


HELP_GPU = """gpu
This field represents the GPU you intend to use for the workload you're
generating. The size of the workload ultimately depends on the capability of
the GPU (i.e. a "small" workload for a powerful GPU may be "large" for a weak
GPU). This field must be one of "gtx0180", "k80", "p100", or "v100"."""

HELP_NUM_JOBS = """num_jobs
The number of jobs in the workload."""

HELP_JOB_MIX = """job_mix
The job_mix represents the mixes of the jobs that are part of a workload.
It must be one of the following values:
  - small
  - medium
  - large
  - random
  - 50
  - 33
  - 25
  - 16
  - v2-90
  - all
They carry the following meanings:
  - small: 100% of the jobs are small
  - medium: 100% of the jobs are medium
  - large: 100% of the jobs are large
  - random: ~33% of the jobs are of each small, medium, or large.
            The jobs from each group are randomly chosen.
            The order of jobs over the entire workload is random.
  - 50: 1/2 of the jobs have a max of 1 - 4 GB memory footprint. The rest have
        a footprint > 4 GB.
  - 33: 1/3 of the jobs have a max of 1 - 4 GB memory footprint. The rest have
        a footprint > 4 GB.
  - 25: 1/4 of the jobs have a max of 1 - 4 GB memory footprint. The rest have
        a footprint > 4 GB.
  - 16: 1/6 of the jobs have a max of 1 - 4 GB memory footprint. The rest have
        a footprint > 4 GB.
  - v2-90: 9/10 of the jobs have < 1 GB memory footprint. The rest have
        a footprint > 1 GB.
  - all: 100% of the jobs are in the workload. This is useful for debugging
         Here, num_jobs is still a required arg, but it's ignored; it'll just
         be equal to the size of all the jobs.
Here, "small", "medium", "large" refer to the max memory footprint
possible for a given job:
  - small: all kernels < 10% GPU memory
  - medium: no large kernels, and at least 1 kernel with 10-50% GPU memory
  - large: at least 1 kernel with > 50% GPU memory"""

def usage_and_exit():
    print()
    print()
    print('Usage:')
    print('    {} <gpu> <job_mix> <num_jobs> [output_filename]'.format(sys.argv[0]))
    print()
    print(HELP_GPU)
    print()
    print(HELP_JOB_MIX)
    print()
    print(HELP_NUM_JOBS)
    print()
    print()
    sys.exit(1)


def parse_args():
    if len(sys.argv) < 4 or len(sys.argv) > 5 :
        usage_and_exit()
    gpu = sys.argv[1]
    if gpu not in GPU_TO_MEM:
        usage_and_exit()
    job_mix = sys.argv[2]
    if job_mix not in {'small', 'medium', 'large', 'random',
                       '50', '33', '25', '16', 'v2-90', 'all'}:
        usage_and_exit()
    num_jobs = sys.argv[3]
    if not num_jobs.isdigit():
        usage_and_exit()
    output_filename = 'example_jobs_00.wl'
    if len(sys.argv) == 5:
        output_filename = sys.argv[4]
    return gpu, job_mix, int(num_jobs), output_filename


def construct_job_arrays(which_gpu):
    print('Constructing the job arrays...')
    total_avail_mem  = GPU_TO_MEM[which_gpu]
    small_threshold  = total_avail_mem * .1
    medium_threshold = total_avail_mem * .5
    for j in all_jobs:
        B = j[0]
        if B < small_threshold:
            small_jobs.append(j[1])
        elif B < medium_threshold:
            medium_jobs.append(j[1])
        else:
            large_jobs.append(j[1])

        if B > (1<<32): # 4 GB
            over_four.append(j[1])
        elif B > (1<<30):
            one_to_four.append(j[1])
        else:
            under_one.append(j[1])

    print('small jobs:')
    for j in small_jobs:
        print('  {}'.format(j))
    print('\nmedium jobs:')
    for j in medium_jobs:
        print('  {}'.format(j))
    print('\nlarge jobs:')
    for j in large_jobs:
        print('  {}'.format(j))
    print('\njobs with < 1 GB footprint:')
    for j in under_one:
        print('  {}'.format(j))
    print('\njobs with 1 - 4 GB footprint:')
    for j in one_to_four:
        print('  {}'.format(j))
    print('\njobs with > 4 GB footprint:')
    for j in over_four:
        print('  {}'.format(j))
    print('\n')



# Generates a workload with 1/3 small, 1/3 medium, and 1/3 large jobs.
# The jobs from each group are randomly chosen.
# The order of jobs over the entire workload is random.
def generate_random_workload():
    workload = []
    random.seed(int(time.time()))
    for i in range(num_jobs):
        # Go to next list of jobs, e.g. small_jobs or medium_jobs
        jobs = job_bufs[i % len(job_bufs)]
        # Pick a job from that list
        job = jobs[random.randint(0, len(jobs)-1)]
        # Append 
        workload.append(job)
    random.shuffle(workload)
    return workload


def generate_proportional_workload():
    D = job_mix_to_denominator[job_mix]
    workload = []
    count = D
    random.seed(int(time.time()))
    for i in range(num_jobs):
        count -= 1
        if count == 0:
            print('M')
            count = D
            job = one_to_four[random.randint(0, len(one_to_four)-1)]
        else:
            print('L')
            job = over_four[random.randint(0, len(over_four)-1)]
        # Append 
        workload.append(job)
    random.shuffle(workload)
    return workload


def generate_v2_proportional_workload():
    D = job_mix_to_denominator[job_mix]
    workload = []
    count = D
    over_one = copy.deepcopy(one_to_four)
    over_one.extend(over_four)
    random.seed(int(time.time()))
    for i in range(num_jobs):
        count -= 1
        if count == 0:
            print('L')
            count = D
            job = over_one[random.randint(0, len(over_one)-1)]
        else:
            print('S')
            job = under_one[random.randint(0, len(under_one)-1)]
        # Append
        workload.append(job)
    random.shuffle(workload)
    return workload

    


def generate_workload():
    workload = []
    jobs = job_mix_to_jobs[job_mix]
    for i in range(num_jobs):
        job = jobs[random.randint(0, len(jobs)-1)]
        workload.append(job)
    return workload


gpu, job_mix, num_jobs, output_filename = parse_args()
construct_job_arrays(gpu)

print('Generating the workload for a "{}" mix...'.format(job_mix))

if job_mix == 'random':
    print('...using generate_random_workload()')
    workload = generate_random_workload()
elif job_mix == 'all':
    print('...using zip()')
    _, workload = zip(*all_jobs)
elif job_mix in ['50', '33', '25', '16']:
    print('...using generate_proporational_workload()')
    workload = generate_proportional_workload()
elif job_mix in ['v2-90']:
    print('...using generate_v2_proporational_workload()')
    workload = generate_v2_proportional_workload()
else:
    print('...using generate_workload()')
    workload = generate_workload()

print()
print('Workload:')
fp = open(output_filename, 'w')
for j in workload:
    full_cmd = RODINIA_BMARK_PATH + '/' + j
    print(full_cmd)
    fp.write(full_cmd+'\n')
fp.close()
print()
print('Workload also written to {}'.format(output_filename))
print()
