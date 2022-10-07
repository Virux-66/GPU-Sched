#!/usr/bin/env python3
import multiprocessing
import subprocess
import sys
import time
import queue
import os
import mmap
import struct





def print_flush(s):
    print(s, flush=True)


def usage_and_exit():
    print()
    print()
    print_flush('Usage:')
    print_flush('    {} <num_processes> <workload_file> <sched_alg>'.format(sys.argv[0]))
    print()
    print_flush('Args:')
    print_flush('  num_processes: The number of worker processes for this driver.')
    print_flush('  workload_file: The relative path and name of the .wl workload file.')
    print_flush('  sched_alg: The scheduling algorithm. One of "zero", "single-assignment", "cg", "mgb_basic", "mgb_simple_compute", or "mgb".')
    print()
    print()
    exit(1)




def run_benchmark(cmd, wid, active_jobs):
    with active_jobs.get_lock():
        active_jobs.value += 1
    # hack for darknet workloads. not sure, but i think using the shell=True
    # for old workloads may not be right, so using this only for the 2 cases
    # where it was needed (where we cat the names of image files to the darknet
    # command)
    if "cat" in cmd and "image-names" in cmd:
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)
    else:
        proc = subprocess.Popen(cmd.split(),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    print_flush('Worker {}: just launched pid {}'.format(wid, proc.pid))
    o, e = proc.communicate()
    rc = proc.returncode
    if rc != 0:
        # XXX Don't change the "got error" string without changing the results
        # parser (and possibly other code to follow). This is what we look for
        # when checking for workloader process errors now.
        print_flush('Worker {} got error: {}'.format(wid, rc))
        print_flush('Worker {} dumping error: {}'.format(wid, e.decode('utf-8')))
    else:
        print_flush('suc')
    try:
        print_flush(o.decode('utf-8'))
        print_flush(e.decode('utf-8'))
    except UnicodeDecodeError as ude:
        print_flush('saw a decode utf-8 error. probably darknet workload? can ignore')
    with active_jobs.get_lock():
        active_jobs.value -= 1


def pool_worker_main(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond):
    print_flush('Pool Worker {}: Starting'.format(wid))
    pool_worker_aux(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond)
    print_flush('Pool Worker {}: Exiting normally'.format(wid))


def pool_worker_aux(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond):
    while True:
        jp_lock.acquire()
        jp_cond.wait()
        jp_lock.release()
        while True:
            next_step = do_work(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond)
            if next_step == 'DONE':
                return
            if next_step == 'WAIT':
                break
            assert next_step == 'CONT'


def worker_main(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond):
    print_flush('Worker {}: Starting'.format(wid))
    while True:
        next_step = do_work(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond)
        if next_step == 'DONE':
            print_flush('Worker {}: Exiting normally'.format(wid))
            return
        elif next_step == 'WAIT':
            break
        assert next_step == 'CONT'
    pool_worker_aux(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond)
    print_flush('Worker {}: Starting'.format(wid))


def adjust_job_pressure(wid, jp_lock, jp_cond):
    read_shm(wid) # XXX Making some multiprocessing assumptions here. This could fail.

    cpu_running_jobs = active_jobs.value - (gpu_running_jobs + gpu_waiting_jobs)
    if cpu_running_jobs > num_processes:
        # This means the CPUs are overloaded
        # Put the worker back in the pool.
        print_flush('Worker {}: C > N. Going back to pool'.format(wid))
        return 'WAIT'
    elif gpu_waiting_jobs >= max_gpu_waiting_jobs:
        # This means the GPU is overloaded, and we're already beyond our
        # threshold queue depth.
        # Put the worker back in the pool
        print_flush('Worker {}: W >= K. Going back to pool'.format(wid))
        return 'WAIT'
    else:
        assert gpu_waiting_jobs < max_gpu_waiting_jobs
        assert cpu_running_jobs <= num_processes
        # This means the CPUs are not overloaded, and we're under our
        # queue depth threshold.
        # Assign a new job to the worker
        # Take a worker from the pool, and assign it a new job
        print_flush('Worker {}: Adding a new worker from the pool and continuing'.format(wid))
        jp_lock.acquire()
        jp_cond.notify()
        jp_lock.release()
        return 'CONT'


def do_work(q, active_jobs, jobs_processed, wid, experiment_start_time, jp_lock, jp_cond):
    try:
        idx, benchmark_cmd = q.get(block=True, timeout=1)
        print_flush('Worker {}: {} {}'.format(wid, idx, benchmark_cmd))
        bmark_start_time = time.time()
        run_benchmark(benchmark_cmd, wid, active_jobs)
        bmark_end_time = time.time()
        bmark_time = bmark_end_time - bmark_start_time
        time_since_start = bmark_end_time - experiment_start_time
        print_flush('Worker {}: TOTAL_BENCHMARK_TIME {} {}'.format(wid, idx, bmark_time))
        print_flush('Worker {}: TIME_SINCE_START     {} {}'.format(wid, idx, time_since_start))
        with jobs_processed.get_lock():
            jobs_processed.value += 1
            if jobs_processed.value == jobs_total:
                experiment_total_time = time.time() - experiment_start_time
                print_flush('Worker {}: TOTAL_EXPERIMENT_TIME {}'.format(wid, experiment_total_time))
                return 'DONE'
            if ENABLE_DYNAMIC_JOB_PRESSURE:
                return adjust_job_pressure(wid, jp_lock, jp_cond)
        # still jobs left. no dynamic job pressure. fall through and return CONT
    except queue.Empty:
        if jobs_processed.value == jobs_total:
            print_flush('Worker {}: Worklist is empty.'.format(wid))
            return 'DONE'
        print_flush('Worker {}: Worklist is empty. Retrying get().'.format(wid))
        print_flush('Worker {}: jobs_processed.value is {}'.format(wid, jobs_processed.value))
    except Exception as e:
        print_flush('Worker {}: Unexpected error when fetching from worklist. ' \
              'Raising.'.format(wid))
        raise
    return 'CONT'




def read_workload_into_q(q, workload_file):
    count = 0
    with open(workload_file) as f:
        for line in f:
            q.put((count, line.strip()))
            count += 1
    return count


def init_shm(fp):
    global shm
    shm = mmap.mmap(fp.fileno(), 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    read_shm(0) # wid == 0... doesn't matter


def read_shm(wid):
    global gpu_running_jobs, gpu_waiting_jobs
    byte_four = shm[12:16]
    byte_five = shm[16:20]
    gpu_running_jobs = struct.unpack('i', byte_four)[0]
    gpu_waiting_jobs = struct.unpack('i', byte_five)[0]
    print_flush('Worker {}: Reading shared memory'.format(wid))
    print_flush('Worker {}: gpu_running_jobs: {}'.format(wid, gpu_running_jobs))
    print_flush('Worker {}: gpu_waiting_jobs: {}'.format(wid, gpu_waiting_jobs))


print_flush('Parsing args')
ENABLE_DYNAMIC_JOB_PRESSURE = False
if len(sys.argv) != 4:
    usage_and_exit()
workload_file = sys.argv[1]
sched_alg     = sys.argv[2]
if sched_alg not in {'zero', 'single-assignment', 'cg', 'mgb', 'mgb_basic', 'mgb_simple_compute', 'mgb'}:
    usage_and_exit()
if sched_alg in {'mgb_basic', 'mgb_simple_compute', 'mgb'}:
    mgb_args = sys.argv[3].split('.')
    if len(mgb_args) == 2:
        max_gpu_waiting_jobs = int(mgb_args[1])
        ENABLE_DYNAMIC_JOB_PRESSURE = True
    num_processes = int(mgb_args[0])
else:
    num_processes = int(sys.argv[3])



print_flush('Starting driver')
print_flush('  workload_file: {}'.format(workload_file))
print_flush('  sched_alg: {}'.format(sched_alg))
print_flush('  num_processes: {}'.format(num_processes))
if ENABLE_DYNAMIC_JOB_PRESSURE:
    print_flush('  ENABLE_DYNAMIC_JOB_PRESSURE: {}'.format(ENABLE_DYNAMIC_JOB_PRESSURE))
    print_flush('  max_gpu_waiting_jobs: {}'.format(max_gpu_waiting_jobs))


q = multiprocessing.Queue()
jp_lock = multiprocessing.Lock()
jp_cond = multiprocessing.Condition(jp_lock)
jobs_processed = multiprocessing.Value('i', 0)
active_jobs = multiprocessing.Value('i', 0)
jobs_total = read_workload_into_q(q, workload_file)
workers = []
pool_workers = []

experiment_start_time = time.time()

if ENABLE_DYNAMIC_JOB_PRESSURE:
    fp_bemps = open('/dev/shm/bemps', 'rb')
    num_pool_processes = num_processes
    shm = None
    gpu_running_jobs = 0
    gpu_waiting_jobs = 0
    init_shm(fp_bemps)
    for i in range(num_pool_processes):
        wid = i + num_processes
        p = multiprocessing.Process(target=pool_worker_main, args=(q,active_jobs,jobs_processed,wid,experiment_start_time,jp_lock,jp_cond,))
        pool_workers.append(p)
        p.start()


for i in range(num_processes):
    p = multiprocessing.Process(target=worker_main, args=(q,active_jobs,jobs_processed,i,experiment_start_time,jp_lock,jp_cond,))
    workers.append(p)
    p.start()

print_flush('Main process: Waiting for workers...')
for w in workers:
    w.join()
print_flush('Main process: Joined all workers.')


if ENABLE_DYNAMIC_JOB_PRESSURE:
    print_flush('Main process: Waiting for pool workers...')
    jp_lock.acquire()
    jp_cond.notify_all()
    jp_lock.release()
    for w in pool_workers:
        w.join()
    print_flush('Main process: Joined all pool workers.')
    shm.close()
    fp_bemps.close()


print_flush('Main process: Exiting normally.')
