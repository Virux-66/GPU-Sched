In this document, the specifications of all benchmarks used in this system are described detailedly. All benchmarks originate from CUDA samples, each of which is either compute-bound or bandwidth-bound.  Due to limited memory of RTX 3080 Ti (12 GB), we set the memory footprint of each benchmark not more than 2 GB such that the system has enought kernels to schedule to explore more parallelism. The operands in all benchmarks are single floating-point and we believe that the model and algorithm we develop are siutable for half floating-point and double floating-point. 

- `${CUDA_HOME}/samples/0_Simple/vectorAdd`
  - data size: num_Elements=100,000,000
  - Launch Stats: 196 blocks of 256 threads.
  - Memory footprint: ~1.2 GB
  - Memory traffic: ~1.2 GB
  - fp operation: 100,000,000

  - arithmetic intensity: 0.0776
  - type: bandwidth-bound



- `${CUDA_HOME}/samples/0_Simple/matrixMul`: when data size is large enough, it is bandwidth-bound.
  - data size: -wA=-hA=-wB=-hB=2048
  - Launch Stats: 4096 blocks of 1024 threads
  - Memory footprint: 48 MB
  - Memory traffic: 902.74 MB
  - fp operation: 17,179,869,184
  - arithmetic intensity: 18.149
  - type: bandwidth-bound



- `${CUDA_HOME}/samples/4_Finance/binomialOptions`
  - data size: `MAX_OPTIONS`= 1024
  - Launch stats: 1024 blocks of 128 threads
  - Memory footprint: ~24 KB
  - Memory traffic:  ~44..4 KB
  - fp operation: 6,511,611,904
  - arithmetic intensity: 143220
  - type: compute-bound



- `${CUDA_HOME}/samples/4_Finance/BlackScholes`
  - data size: `OPT_N` = 80,000,000
  - Launch stats: 312500 blocks of 128 threads
  - Memory footprint: ~1.6 GB
  - Memory traffic: ~1.6 GB
  - fp operation: 5,751,322,851
  - arithmetic intensity: 3.3477
  - type: bandwidth-bound



- `${CUDA_HOME}/samples/6_Advanced/scan`: `scanExclusiveShared`, `scanExclusiveShared2`, `uniformUpdate`

  - data size: 208 arrays of 262144 elements

  - Launch stats: 53248 blocks of 256 threads
  - Memory footprint: 416.25 MB
  - Memory traffic: 433.04 MB + 3.50 MB + 435.85 MB = 872.39 MB
  - fp operation: 0
  - arithmetic intensity: 0
  - type: bandwidth-bound

  

- `${CUDA_HOME}/samples/3_Imaging/histogram`: `histogram256Kernel`, `mergeHistogram256Kernel`
  - data size:  byteCount = 67,108,864
  - Launch stats: 240 blocks of 192 threads
  - Memory footprint: 512.23 MB
  - Memory traffic: 537.22 MB
  - fp operation: 0
  - arithmetic intensity: 0
  - type: bandwidth-bound
