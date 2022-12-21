Each workload, either 16 benchmarks or 32 benchmarks, consists of different ratio of compute-bound and memory-bound benchmarks.
The ratio could be as follow:
- compute-bound : memory-bound = 1 : 1
- compute-bound : memory-bound = 3 : 1
- compute-bound : memory-bound = 5 : 3
- compute-bound : memory-bound = 7 : 1
There four ratios are identified with suffix starting from 0 to 3.
Take `3080Ti_32jobs_0.wl` as example, it indicates this workload includes 32 benchmarks,
which consists of 1/2 compute-bound and 1/2 memory-bound and it runs on 3080Ti.