In this document, the specifications of all benchmarks used in this system are described detailedly. All benchmarks originate from CUDA samples, each of which is either compute-bound or bandwidth-bound.  Due to limited memory of RTX 3080 Ti (12 GB), we set the memory footprint of each benchmark not more than 2 GB such that the system has enought kernels to schedule to explore more parallelism. The operands in all benchmarks are single floating-point and we believe that the model and algorithm we develop are siutable for half floating-point and double floating-point. 
**`block-level` means that Memory traffic, fp operation and arithmetic intensity having been measured are in thread block level.** In future, we probably compare the evaluation results of these two different granularity-level.

- `${CUDA_HOME}/samples/0_Simple/vectorAdd`
  - data size: num_Elements=100,000,000
  - Launch Stats: 196 blocks of 256 threads.
  - Memory footprint: ~1.2 GB
  - Memory traffic: 5.63 KB
  - fp operation: 256
  - arithmetic intensity: 0.0444
  - type: bandwidth-bound


- `${CUDA_HOME}/samples/0_Simple/matrixMul`: when data size is large enough, it is bandwidth-bound.
  - data size: -wA=-hA=-wB=-hB=2048
  - Launch Stats: 4096 blocks of 1024 threads
  - Memory footprint: 48 MB
  - Memory traffic: 530.18 KB
  - fp operation: 4,194,304
  - arithmetic intensity: 7.7256
  - type: bandwidth-bound


  `${CUDA_HOME}/samples/0_Simple/matrixMul`:
  - data size: -wA=-hA=-wB=-hB=1024
  - Launch Stats: 1024 blocks of 1024 threads 
  - Memory footprint: 12 MB
  - Memory traffic: 267.78 KB
  - fp operation: 2,097,152
  - arithmetic intensity: 7.6408
  - type: bandwidth-bound


- `${CUDA_HOME}/samples/4_Finance/binomialOptions`
  - data size: `MAX_OPTIONS`= 1024
  - Launch stats: 1024 blocks of 128 threads
  - Memory footprint: ~24 KB
  - Memory traffic:  17.66 KB
  - fp operation: 6,358,996
  - arithmetic intensity: 351.6396
  - type: compute-bound


- `${CUDA_HOME}/sample/4_Finance/binomialOptions`
  - data size: `MAX_OPTIONS`=64
  - Launch stats: 64 blocks of 128 threads 
  - Memory footprint: 24 B
  - Memory traffic: 17.66 KB
  - fp operation: 6,358,996
  - arithmetic intensity: 351.6396
  - type: compute-bound


- `${CUDA_HOME}/samples/4_Finance/BlackScholes`
  - data size: `OPT_N` = 80,000,000
  - Launch stats: 312500 blocks of 128 threads
  - Memory footprint: ~1.6 GB
  - Memory traffic: 15.74 KB
  - fp operation: 16,900
  - arithmetic intensity: 1.0485
  - type: bandwidth-bound


- `${CUDA_HOME}/samples/6_Advanced/scan`: `scanExclusiveShared`, `scanExclusiveShared2`, `uniformUpdate`
  - data size: 208 arrays of 262144 elements
  - Launch stats: 53248 blocks of 256 threads
  - Memory footprint: 416.25 MB
  - Memory traffic: 8.32 KB + 20.61 KB + 8.19 KB = 37.12 KB
  - fp operation: 0
  - arithmetic intensity: 0
  - type: bandwidth-bound


- `${CUDA_HOME}/samples/3_Imaging/histogram`: `histogram256Kernel`, `mergeHistogram256Kernel`
  - data size:  byteCount = 67,108,864
  - Launch stats: 240 blocks of 192 threads
  - Memory footprint: 512.23 MB
  - Memory traffic: 545.79 MB + 14.46 KB = 545.80 MB
  - fp operation: 0
  - arithmetic intensity: 0
  - type: bandwidth-bound
