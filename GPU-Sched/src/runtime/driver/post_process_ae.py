#!/usr/bin/env python3
import sys
import statistics
import re




BASE_PATH     = 'results'
BASE_PATH_REF = 'results_reference_ae'
CG_CRASH_LOGS_PATH = BASE_PATH_REF + '/table-2'
NVPROF_LOGS_PATH   = BASE_PATH_REF + '/table-4'

SCHED_LOG_SUF  = 'sched-log'
SCHED_STAT_SUF = 'sched-stats'
WRKLDR_LOG_SUF = 'workloader-log'
GPU_UTIL_SUF   = 'sched_gpu.csv'




def usage_and_exit():
    print()
    print('  Usage:')
    print('    {} <figure-or-table> [--ref]'.format(sys.argv[0]))
    print()  
    print('    If providing a figure, it must be one of:')
    print('      {figure-4, figure-5, figure-6, figure-7, figure-8}')
    print('    If providing a table, it must be one of:')
    print('      {table-2, table-3, table-4}')
    print()
    print('    The --ref switch is optional. It will use the reference logs,')
    print('    which are the same logs used in the paper. Note that table-2')
    print('    and table-4 always use the reference logs (i.e. the switch')
    print('    has no effect).')
    print()
    sys.exit(1)


def format_rodinia_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'single-assignment.4', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'cg.6', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.16', WRKLDR_LOG_SUF),
    )


def format_rodinia_util_filenames(workload):
    return (
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'single-assignment.4', GPU_UTIL_SUF),
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'cg.6', GPU_UTIL_SUF),
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'mgb_basic.16', GPU_UTIL_SUF),
    )


def format_rodinia_alg_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb.16', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.16', WRKLDR_LOG_SUF),
    )


def format_rodinia_nvprof_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(NVPROF_LOGS_PATH, workload, 'single-assignment.4', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(NVPROF_LOGS_PATH, workload, 'mgb.16', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(NVPROF_LOGS_PATH, workload, 'mgb_basic.16', WRKLDR_LOG_SUF),
    )


def format_darknet_filenames(workload):
    return (
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'zero.8', WRKLDR_LOG_SUF),
        '{}/{}.{}.{}'.format(BASE_PATH, workload, 'mgb_basic.8', WRKLDR_LOG_SUF),
    )


def format_darknet_util_filenames(workload):
    return (
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'zero.8', GPU_UTIL_SUF),
        '{}/{}.{}-{}'.format(BASE_PATH, workload, 'mgb_basic.8', GPU_UTIL_SUF),
    )


def format_crash_filename(workload):
    return '{}/{}.{}'.format(CG_CRASH_LOGS_PATH, workload, WRKLDR_LOG_SUF)


def convert_time_to_ms(t):
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
    return v


def calc_average_kernel_times(cmd_to_invocations, avgs, which):
    for cmd, invocations in cmd_to_invocations.items():
        for invocation in invocations:
            for kernel_name, t in invocation.items():
                k = cmd + kernel_name
                if k not in avgs[which]:
                    avgs[which][k] = 0
                avgs[which][k] += t
        avgs[which][k] = 1.0 * avgs[which][k] / len(invocations)


def post_process_figure_4():
    print()
    print('Post-process Figure 4')
    print()
    alg2_throughputs = []
    alg3_throughputs = []
    for workload in workloads_rodinia:
        alg2_filename, alg3_filename = format_rodinia_alg_filenames(workload)

        alg2_throughput, _   = parse_workloader_log(alg2_filename)
        alg3_throughput, _   = parse_workloader_log(alg3_filename)

        alg2_throughputs.append(alg2_throughput)
        alg3_throughputs.append(alg3_throughput)

    alg3_throughput_improvements = [ t[1] / t[0] for t in zip(alg2_throughputs, alg3_throughputs) ]
    avg_alg3_throughput_improvement = statistics.mean(alg3_throughput_improvements)

    print('THROUGHPUT IMPROVEMENT')
    print('. alg2 alg3')
    for idx, workload in enumerate(workloads_rodinia):
        name = rodinia_workloads_to_abbr_name[workload]
        print('{} {} {}'.format(name, 1, round(alg3_throughput_improvements[idx], 2)))
    print('{} {} {}'.format('average', 1, round(avg_alg3_throughput_improvement, 2)))
    print()

    print('SUMMARY')
    print('Avg Alg3 Throughput Improvement: {}'.format(round(avg_alg3_throughput_improvement, 2)))
    print()


def post_process_figure_5():
    print()
    print('Post-process Figure 5')
    print()
    sa_throughputs  = []
    cg_throughputs  = []
    mgb_throughputs = []
    for workload in workloads_rodinia:
        sa_filename, cg_filename, mgb_filename = format_rodinia_filenames(workload)

        sa_throughput, _   = parse_workloader_log(sa_filename)
        cg_throughput, _   = parse_workloader_log(cg_filename)
        mgb_throughput, _  = parse_workloader_log(mgb_filename)

        sa_throughputs.append(sa_throughput)
        cg_throughputs.append(cg_throughput)
        mgb_throughputs.append(mgb_throughput)

    cg_throughput_improvements  = [ t[1] / t[0] for t in zip(sa_throughputs, cg_throughputs) ]
    mgb_throughput_improvements = [ t[1] / t[0] for t in zip(sa_throughputs, mgb_throughputs) ]
    avg_cg_throughput_improvement  = statistics.mean(cg_throughput_improvements)
    avg_mgb_throughput_improvement = statistics.mean(mgb_throughput_improvements)

    print('THROUGHPUT IMPROVEMENT')
    print('. sa cg case')
    for idx, workload in enumerate(workloads_rodinia):
        name = rodinia_workloads_to_abbr_name[workload]
        print('{} {} {} {}'.format(name, 1, round(cg_throughput_improvements[idx],2),
                                  round(mgb_throughput_improvements[idx],2)))
    print('{} {} {} {}'.format('average', 1, round(avg_cg_throughput_improvement,2),
                                  round(avg_mgb_throughput_improvement,2)))
    print()

    print('SUMMARY')
    print('Avg CG   Throughput Improvement: {}'.format(round(avg_cg_throughput_improvement,2)))
    print('Avg CASE Throughput Improvement: {}'.format(round(avg_mgb_throughput_improvement,2)))
    print()


def post_process_figure_6():
    print()
    print('Post-process Figure 6')
    print()
    sa_throughputs  = []
    cg_throughputs  = []
    mgb_throughputs = []
    workload = 'v100_33_32jobs_0'
    sa_filename, cg_filename, mgb_filename = format_rodinia_util_filenames(workload)
    
    sa_peak_utilization, sa_avg_utilization  = parse_utilization_log(sa_filename)
    cg_peak_utlization, cg_avg_utilization  = parse_utilization_log(cg_filename)
    mgb_peak_utilization, mgb_avg_utilization = parse_utilization_log(mgb_filename)

    print('PEAK UTILIZATION')
    print('SA   : {}'.format(round(sa_peak_utilization, 2)))
    print('CG   : {}'.format(round(cg_peak_utlization, 2)))
    print('CASE : {}'.format(round(mgb_peak_utilization, 2)))
    print()
    print('AVG UTILIZATION')
    print('SA   : {}'.format(round(sa_avg_utilization, 2)))
    print('CG   : {}'.format(round(cg_avg_utilization, 2)))
    print('CASE : {}'.format(round(mgb_avg_utilization, 2)))
    print()


def post_process_figure_7():
    print()
    print('Post-process Figure 7')
    print()
    zero_throughputs  = []
    mgb_throughputs = []
    for workload in workloads_darknet:
        zero_filename, mgb_filename = format_darknet_filenames(workload)

        zero_throughput, _   = parse_workloader_log(zero_filename)
        mgb_throughput, _    = parse_workloader_log(mgb_filename)

        zero_throughputs.append(zero_throughput)
        mgb_throughputs.append(mgb_throughput)

    mgb_throughput_improvements = [ t[1] / t[0] for t in zip(zero_throughputs, mgb_throughputs) ]
    avg_mgb_throughput_improvement = statistics.mean(mgb_throughput_improvements)

    print('THROUGHPUT IMPROVEMENT')
    print('. schedgpu case')
    for idx, workload in enumerate(workloads_darknet):
        nn_task = darknet_workload_to_nn_task[workload]
        print('{} {} {}'.format(nn_task, 1, round(mgb_throughput_improvements[idx],2)))
    print('{} {} {}'.format('average', 1, round(avg_mgb_throughput_improvement,2)))
    print()

    print('SUMMARY')
    print('Avg CASE Throughput Improvement: {}'.format(round(avg_mgb_throughput_improvement,2)))
    print()


def post_process_figure_8():
    print()
    print('Post-process Figure 8')
    print()
    zero_throughputs  = []
    mgb_throughputs = []
    workload = 'v100_0_8jobs_3'
    zero_filename, mgb_filename = format_darknet_util_filenames(workload)
    
    zero_peak_utilization, zero_avg_utilization  = parse_utilization_log(zero_filename)
    mgb_peak_utilization, mgb_avg_utilization = parse_utilization_log(mgb_filename)

    print('PEAK UTILIZATION')
    print('SchedGPU: {}'.format(round(zero_peak_utilization, 2)))
    print('CASE:     {}'.format(round(mgb_peak_utilization, 2)))
    print()
    print('AVG UTILIZATION')
    print('SchedGPU: {}'.format(round(zero_avg_utilization, 2)))
    print('CASE:     {}'.format(round(mgb_avg_utilization, 2)))
    print()


def post_process_table_2():
    print()
    print('Post-process Table 2')
    print()

    workload_to_crashes = {}
    for workload in workloads_crash:
        filename = format_crash_filename(workload)
        workload_to_crashes[workload] = 0
        with open(filename) as f:
            for line in f:
                if "got error" in line:
                    workload_to_crashes[workload] += 1

    # iterate over number of workers used for the CG experiment (6, 8, etc.)
    crashes = {}
    for n in ['6', '8', '10', '12']:
        crashes[n] = []
        # iterater over the mixes (1:1, 2:1, 3:1, 5:1)
        for m in ['50', '33', '25', '16']:
            has_crash = 0
            if workload_to_crashes['v100_'+m+'_16jobs_0.cg.'+n]:
                has_crash += 1
            if workload_to_crashes['v100_'+m+'_32jobs_0.cg.'+n]:
                has_crash += 1
            p = workload_to_crashes['v100_'+m+'_16jobs_0.cg.'+n] \
                + workload_to_crashes['v100_'+m+'_32jobs_0.cg.'+n]
            if p:
                # Table 2 reports result for the following question:
                # In cases where there's at least 1 crash, what percent of
                # those CG workers crashed?
                p = round(p/(int(n)*has_crash), 2)
            crashes[n].append(str(p))
    print('num_workers 1:1mix 2:1mix 3:1mix 5:1mix')
    print('6 {}'.format(' '.join(crashes['6'])))
    print('8 {}'.format(' '.join(crashes['8'])))
    print('10 {}'.format(' '.join(crashes['10'])))
    print('12 {}'.format(' '.join(crashes['12'])))
    print()


def post_process_table_3():
    print()
    print('Post-process Table 3')
    print()
    sa_job_turnarounds  = []
    cg_job_turnarounds  = []
    mgb_job_turnarounds = []
    for workload in workloads_rodinia:
        sa_filename, cg_filename, mgb_filename = format_rodinia_filenames(workload)

        _, sa_times_since_start   = parse_workloader_log(sa_filename)
        _, cg_times_since_start   = parse_workloader_log(cg_filename)
        _, mgb_times_since_start  = parse_workloader_log(mgb_filename)

        sa_job_turnarounds.append(sa_times_since_start)
        cg_job_turnarounds.append(cg_times_since_start)
        mgb_job_turnarounds.append(mgb_times_since_start)

    mgb_turnaround_speedups = []
    assert len(workloads_rodinia) == len(sa_job_turnarounds)
    assert len(workloads_rodinia) == len(mgb_job_turnarounds)
    for idx, workload in enumerate(workloads_rodinia):
        workload_turnaround_speedups = [ t[0] / t[1] for t in zip(sa_job_turnarounds[idx], mgb_job_turnarounds[idx]) ]
        mgb_turnaround_speedups.append(statistics.mean(workload_turnaround_speedups))
    avg_mgb_turnaround_speedup = statistics.mean(mgb_turnaround_speedups)

    print('TURNAROUND SPEEDUP')
    print('. sa case')
    for idx, workload in enumerate(workloads_rodinia):
        name = rodinia_workloads_to_name[workload]
        print('{} {} {}'.format(name, 1, round(mgb_turnaround_speedups[idx],1)))
    print('{} {} {}'.format('average', 1, round(avg_mgb_turnaround_speedup,1)))
    print()


    print('SUMMARY')
    print('Avg CASE Turnaround Speedup: {}'.format(round(avg_mgb_turnaround_speedup,1)))
    print()

    print()


def post_process_table_4():
    print()
    print('Post-process Table 4')
    print()

    alg2_slowdowns = []
    alg3_slowdowns = []
    for workload in workloads_rodinia_nvprof:
        cmd_to_invocations_sa   = {}
        cmd_to_invocations_alg2 = {}
        cmd_to_invocations_alg3 = {}

        sa_filename, alg2_filename, alg3_filename = format_rodinia_nvprof_filenames(workload)

        parse_nvprof_log(sa_filename, cmd_to_invocations_sa)
        parse_nvprof_log(alg2_filename, cmd_to_invocations_alg2)
        parse_nvprof_log(alg3_filename, cmd_to_invocations_alg3)

        for k in cmd_to_invocations_sa:
            assert k in cmd_to_invocations_alg2
            assert k in cmd_to_invocations_alg3
        for k in cmd_to_invocations_alg2:
            assert k in cmd_to_invocations_sa
            assert k in cmd_to_invocations_alg3
        for k in cmd_to_invocations_alg3:
            assert k in cmd_to_invocations_sa
            assert k in cmd_to_invocations_alg2

        SA_AVGS   = 0
        ALG2_AVGS = 1
        ALG3_AVGS = 2
        avgs = [{}, {}, {}]

        calc_average_kernel_times(cmd_to_invocations_sa,   avgs, SA_AVGS)
        calc_average_kernel_times(cmd_to_invocations_alg2, avgs, ALG2_AVGS)
        calc_average_kernel_times(cmd_to_invocations_alg3, avgs, ALG3_AVGS)

        alg2_slowdowns_kernels = []
        alg3_slowdowns_kernels = []

        for k in avgs[SA_AVGS]:
            sa_avg_kernel_time   = avgs[SA_AVGS][k]
            alg2_avg_kernel_time = avgs[ALG2_AVGS][k]
            alg3_avg_kernel_time = avgs[ALG3_AVGS][k]
            alg2_slowdown = 1.0 * alg2_avg_kernel_time / sa_avg_kernel_time
            alg3_slowdown = 1.0 * alg3_avg_kernel_time / sa_avg_kernel_time
            if '[CUDA memcpy DtoH]' in k or '[CUDA memcpy HtoD]' in k:
                pass
            else:
                alg2_slowdowns_kernels.append(alg2_slowdown)
                alg3_slowdowns_kernels.append(alg3_slowdown)

        alg2_avg_kernel_slowdown = statistics.mean(alg2_slowdowns_kernels)
        alg3_avg_kernel_slowdown = statistics.mean(alg3_slowdowns_kernels)
        alg2_avg_kernel_slowdown = round((alg2_avg_kernel_slowdown - 1) * 100, 1)
        alg3_avg_kernel_slowdown = round((alg3_avg_kernel_slowdown - 1) * 100, 1)
        alg2_slowdowns.append(alg2_avg_kernel_slowdown)
        alg3_slowdowns.append(alg3_avg_kernel_slowdown)
    # append avg
    alg2_slowdowns.append(round(statistics.mean(alg2_slowdowns), 1))
    alg3_slowdowns.append(round(statistics.mean(alg3_slowdowns), 1))

    print('KERNEL SLOWDOWN')
    print('. W1 W2 W3 W4 W5 W6 W7 W8 Avg')
    print('Alg2 {}'.format(' '.join(str(x) for x in alg2_slowdowns)))
    print('Alg3 {}'.format(' '.join(str(x) for x in alg3_slowdowns)))
    print()


def parse_workloader_log(filename):
    #
    # Examples of what we're looking for:
    #
    # time since start:
    #   Worker 4: TIME_SINCE_START     4 9.485850095748901
    # bmark times:
    #   Worker 0: TOTAL_BENCHMARK_TIME 0 2.99345064163208
    #   Worker 1: TOTAL_BENCHMARK_TIME 2 1.2312407493591309
    # total time:
    #   Worker 0: TOTAL_EXPERIMENT_TIME 4.400961637496948
    #
    time_since_start = {}
    bmark_times = []
    total_time  = 0
    throughput = 0
    with open(filename) as f:
        for line in f:
            if 'TOTAL_BENCHMARK_TIME' in line:
                line = line.strip().split()
                bmark_times.append( (int(line[3]), float(line[4])) )
            elif 'TOTAL_EXPERIMENT_TIME' in line:
                total_time = float(line.strip().split()[3])
            elif 'TIME_SINCE_START' in line:
                time_since_start[float(line.strip().split()[3])] = float(line.strip().split()[4])

    throughput = len(bmark_times) / float(total_time)

    # grab just the sorted-by-worker time_since_start values
    # (k is the worker idx. v is its time since start)
    time_since_start = [v for k, v in sorted(time_since_start.items())]

    return throughput, time_since_start


def parse_utilization_log(filename):
    utilizations = []
    peak_utilization = 0
    with open(filename) as f:
        next(f) # skip header
        for line in f:
            line_vec = line.strip().split(',')
            timestamp = line_vec[0]
            device_0  = int(line_vec[1])
            device_1  = int(line_vec[2])
            device_2  = int(line_vec[3])
            device_3  = int(line_vec[4])
            utilization = device_0 + device_1 + device_2 + device_3
            if utilization > peak_utilization:
                peak_utilization = utilization
            utilization = utilization / 4 / 100
            utilizations.append(utilization)
    peak_utilization = peak_utilization / 4 /100
    avg_utilization = statistics.mean(utilizations)
    return peak_utilization, avg_utilization


def parse_nvprof_log(workload_log, cmd_to_invocations):
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




post_process_funcs = {
    'figure-4': post_process_figure_4,
    'figure-5': post_process_figure_5,
    'figure-6': post_process_figure_6,
    'figure-7': post_process_figure_7,
    'figure-8': post_process_figure_8,
    'table-2': post_process_table_2,
    'table-3': post_process_table_3,
    'table-4': post_process_table_4,
}

workloads_rodinia = [
    'v100_50_16jobs_0',
    'v100_33_16jobs_0',
    'v100_25_16jobs_0',
    'v100_16_16jobs_0',
    'v100_50_32jobs_0',
    'v100_33_32jobs_0',
    'v100_25_32jobs_0',
    'v100_16_32jobs_0',
]

rodinia_workloads_to_name = {
    'v100_50_16jobs_0': '16-job-1:1-mix',
    'v100_33_16jobs_0': '16-job-2:1-mix',
    'v100_25_16jobs_0': '16-job-3:1-mix',
    'v100_16_16jobs_0': '16-job-5:1-mix',
    'v100_50_32jobs_0': '32-job-1:1-mix',
    'v100_33_32jobs_0': '32-job-2:1-mix',
    'v100_25_32jobs_0': '32-job-3:1-mix',
    'v100_16_32jobs_0': '32-job-5:1-mix',
}

rodinia_workloads_to_abbr_name = {
    'v100_50_16jobs_0': 'W1',
    'v100_33_16jobs_0': 'W2',
    'v100_25_16jobs_0': 'W3',
    'v100_16_16jobs_0': 'W4',
    'v100_50_32jobs_0': 'W5',
    'v100_33_32jobs_0': 'W6',
    'v100_25_32jobs_0': 'W7',
    'v100_16_32jobs_0': 'W8',
}

workloads_rodinia_nvprof = [
    'v100_50_16jobs_1',
    'v100_33_16jobs_1',
    'v100_25_16jobs_1',
    'v100_16_16jobs_1',
    'v100_50_32jobs_1',
    'v100_33_32jobs_1',
    'v100_25_32jobs_1',
    'v100_16_32jobs_1',
]

workloads_darknet = [
    'v100_2_8jobs_0',
    'v100_0_8jobs_1',
    'v100_0_8jobs_2',
    'v100_0_8jobs_3',
]

darknet_workload_to_nn_task = {
    'v100_2_8jobs_0': 'predict',
    'v100_0_8jobs_1': 'detect',
    'v100_0_8jobs_2': 'generate',
    'v100_0_8jobs_3': 'train',
}

workloads_crash = [
    'v100_16_16jobs_0.cg.10',
    'v100_16_16jobs_0.cg.12',
    'v100_16_16jobs_0.cg.6',
    'v100_16_16jobs_0.cg.8',
    'v100_16_32jobs_0.cg.10',
    'v100_16_32jobs_0.cg.12',
    'v100_16_32jobs_0.cg.6',
    'v100_16_32jobs_0.cg.8',
    'v100_25_16jobs_0.cg.10',
    'v100_25_16jobs_0.cg.12',
    'v100_25_16jobs_0.cg.6',
    'v100_25_16jobs_0.cg.8',
    'v100_25_32jobs_0.cg.10',
    'v100_25_32jobs_0.cg.12',
    'v100_25_32jobs_0.cg.6',
    'v100_25_32jobs_0.cg.8',
    'v100_33_16jobs_0.cg.10',
    'v100_33_16jobs_0.cg.12',
    'v100_33_16jobs_0.cg.6',
    'v100_33_16jobs_0.cg.8',
    'v100_33_32jobs_0.cg.10',
    'v100_33_32jobs_0.cg.12',
    'v100_33_32jobs_0.cg.6',
    'v100_33_32jobs_0.cg.8',
    'v100_50_16jobs_0.cg.10',
    'v100_50_16jobs_0.cg.12',
    'v100_50_16jobs_0.cg.6',
    'v100_50_16jobs_0.cg.8',
    'v100_50_32jobs_0.cg.10',
    'v100_50_32jobs_0.cg.12',
    'v100_50_32jobs_0.cg.6',
    'v100_50_32jobs_0.cg.8',
]


if len(sys.argv) != 2 and len(sys.argv) != 3:
    usage_and_exit()
if sys.argv[1] not in post_process_funcs:
    usage_and_exit()

if len(sys.argv) == 3:
    if sys.argv[2] != '--ref':
        usage_and_exit()
    BASE_PATH = BASE_PATH_REF + '/' + sys.argv[1]

post_process_funcs[sys.argv[1]]()
