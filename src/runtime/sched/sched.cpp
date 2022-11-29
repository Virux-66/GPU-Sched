#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <ctype.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <list>
#include <queue>

#include <libstatus/profiler.h>

#include "bemps.hpp"
//the library for solving integer linear programming.
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/sat/cp_model_solver.h"
#include <memory>

//#define BEMPS_SCHED_DEBUG

#define SCHED_DEFAULT_BATCH_SIZE 1
#define SCHED_VECTOR_BATCH_SIZE 10
#define SCHED_MGB_BATCH_SIZE    10

// jobs that we allow onto each GPU before checking the threshold.
#define SCHED_MGB_SIMPLE_COMPUTE_MIN_JOBS  (1)
// after min_jobs is exceeded, we check:
//   (active warps + new jobs' warps) / max warps
// --which must be less than this threshold for the new job to get scheduled
#define SCHED_MGB_SIMPLE_COMPUTE_THRESHOLD (0.9f)


// XXX
// Deprecated specs. We use cudaGetDeviceProperties now.

#define GTX_1080_SPECS {            \
  .mem_B = 8116L * 1024 * 1024,     \
  .cores = 2560,                    \
  .num_sms              = 20,       \
  .thread_blocks_per_sm = 32,       \
  .warps_per_sm         = 64,       \
  .total_thread_blocks  = 20 * 32,  \
  .total_warps          = 20 * 64   \
}
#define P100_PCIE_SPECS {           \
  .mem_B = 14000L * 1024 * 1024,    \
  .cores = 3584,                    \
  .num_sms              = 60,       \
  .thread_blocks_per_sm = 32,       \
  .warps_per_sm         = 64,       \
  .total_thread_blocks  = 60 * 32,  \
  .total_warps          = 60 * 64   \
}
#define V100_SXM2_SPECS {           \
  .mem_B = 10000L * 1024 * 1024,    \
  .cores = 5120,                    \
  .num_sms              = 80,       \
  .thread_blocks_per_sm = 32,       \
  .warps_per_sm         = 64,       \
  .total_thread_blocks  = 80 * 32,  \
  .total_warps          = 80 * 64   \
}
#define RTX_3080Ti_SPECS {          \
  .mem_B = 12288L * 1024 * 1024,    \
  .cores = 10240,                   \
  .num_sms              = 80,       \
  .thread_blocks_per_sm = 16,       \
  .warps_per_sm         = 48,       \       
  .total_thread_blocks  = 80 * 16,  \
  .total_warps          = 80 * 48   \
}

#define RTX_3060_SPECS {            \
  .mem_B = 12288L * 1024 * 1024,    \
  .cores = 3584,                    \
  .num_sms              = 28,       \
  .thread_blocks_per_sm = 16,       \
  .warps_per_sm         = 48,       \
  .total_thread_blocks  = 28 * 16,  \
  .total_warps          = 28 * 48   \
}

// FIXME
// kepler, maxwell, pascal, volta all have 64 warps per sm. not clear how to
// get it with cudaGetDeviceProperties, so we're hard-coding it for now.
// Reference: "Compute Capability" section in the nvidia tesla v100 gpu
// architecture document (page 18 in the document, page 23 for adobe).
// Thread blocks per sm is a similar issue.
// There's a discrepancy between cuda versions on sloppyjoe and AWS probably.
// prop.maxBlocksPerMultiProcessor works on sloppyjoe but not in AWS.
// But this is always 32 for our cases, so leaving as a #define, as well.
// MEM_B is a separate issue but still part of this problem of making this
// generic via cudaGetDeviceProperties. We need padding still, apparently due
// to that device0 300MB structure or whatever is going on. So those
// adjusted max mem_B values are defined here, as well, and used in place
// of the generic value given by cudaGetDeviceProperties. (Could try to
// use a percentage of max in order to make it more generic, but that's
// not tested yet.)

//#define WARPS_PER_SM 64 //for 3080Ti, the maximum warp per SM is 48, as with V100, P100 and 1080 the number is 64
#define WARP_SIZE    32  //for 3080Ti and 3060
#define WARPS_PER_SM 48
//#define THREAD_BLOCKS_PER_SM 32 //for any other mentioned GPU other than 3080Ti, the value is 32.
#define THREAD_BLOCKS_PER_SM 16
#define GTX_1080_SPECS_MEM_B (8116L * 1024 * 1024)
//#define V100_SXM2_SPECS_MEM_B (10000L * 1024 * 1024)
//#define V100_SXM2_SPECS_MEM_B (13000L * 1024 * 1024)
#define V100_SXM2_SPECS_MEM_B (14000L * 1024 * 1024)
#define RTX_3080Ti_SEPCS_MEM_B (12288L * 1024 * 1024)
#define RTX_3060_SPECS_MEM_B   (12288L * 1024 * 1024)

//repective arithmetic intensity that can make kernels achieve ridge point
//these values are not supposed to write in to the structure gpu_s.
//Beacause for the same version with different memory probably come with different bandwidth.
//So for a scheduler program used in practice, this value should be measured before
//scheduler starts.
#define RTX_3060_SPECS_AI_FP32_DRAM     0
#define RTX_3060_SPECS_AI_FP32_L1       0
#define RTX_3080Ti_SPECS_AI_FP32_DRAM  (37.736f)
#define RTX_3080Ti_SPECS_AI_FP32_L1    (16.484f)

#ifdef BEMPS_SCHED_DEBUG
#define BEMPS_SCHED_LOG(str)                                          \
  do {                                                                \
    std::cout << get_time_ns() << " " << __FUNCTION__ << ": " << str; \
    std::cout.flush();                                                \
  } while (0)
#else
#define BEMPS_SCHED_LOG(str) \
  do {                       \
  } while (0)
#endif


const int SCHED_ALIVE_COUNT_MAX = 30; // roughly 5s, assuming 100ms timer and 0 beacons
int SCHED_ALIVE_COUNT = 0;
#define ALIVE_MSG()                              \
  do {                                                \
    if (SCHED_ALIVE_COUNT == SCHED_ALIVE_COUNT_MAX) { \
      BEMPS_SCHED_LOG("alive\n");                     \
      SCHED_ALIVE_COUNT = 0;                          \
    } else {                                          \
      SCHED_ALIVE_COUNT++;                            \
    }                                                 \
  } while(0)


#define SCHED_NUM_STOPWATCHES 4   //FIX: replaced by 4
typedef enum {//timing type
 // time the scheduler spends awake and processing
  SCHED_STOPWATCH_AWAKE = 0,

  // time spent in allocate_compute() on success
  SCHED_STOPWATCH_ALLOCATE_COMPUTE_SUCCESS,

  // time spent in allocate_compute() when resources aren't available
  SCHED_STOPWATCH_ALLOCATE_COMPUTE_FAIL,

  // time spent in decision-making of integer linear programming
  SCHED_STOPWATCH_DECISION_MAKING
} sched_stopwatch_e;


typedef enum {
  SCHED_ALG_ZERO_E = 0,
  //SCHED_ALG_ROUND_ROBIN_E,
  //SCHED_ALG_ROUND_ROBIN_BEACONS_E,
  //SCHED_ALG_VECTOR_E,
  SCHED_ALG_SINGLE_ASSIGNMENT_E,
  SCHED_ALG_CG_E,
  SCHED_ALG_MGB_BASIC_E,           // mgb from original ppopp21 submission
  //SCHED_ALG_MGB_SIMPLE_COMPUTE_E,  // mgb simple compute
  SCHED_ALG_MGB_E,     // mgb with SM scheduler emulation
  SCHED_ALG_AI_E       // heuristic algorithm based on arithmetic intensity
} sched_alg_e;

typedef enum{
  SOLVE_ALG_ZERO_E=0
} solve_alg_e;

struct gpu_s {
  long mem_B;
  unsigned int cores;
  unsigned int num_sms;
  unsigned int thread_blocks_per_sm;
  unsigned int warps_per_sm;
  unsigned int total_thread_blocks;
  unsigned int total_warps;
};

struct gpu_in_use_s {
  /*
  In sched_ai_heuristic, only if active_jobs is equal to,
  scheduler can issue concurent set to GPU.
  */
  unsigned int active_jobs;
  /*
  1 if 1 job saturated the compute units. else 0. 
  saturated doesn't indicate one task make use of all compute resource of GPU and thus, 
  no any other tasks can be issued to this GPU. 
  This member is used to quick allocation which does't account each assignment of warps to SMs. 
  In sched_ai_heuristic algorithm, compute_saturated =1 indicates there is concurrent set in a specific GPU
  Once a concurrent set is issued, compute_saturated should be set to 1.
  */
  int compute_saturated; 
  long mem_B; //  mem_B IN USE
  long warps; //  warps IN USE
  int curr_sm; // the next streaming multiprocessor to assign
  std::vector<std::pair<int, int>> *sms; // (thread blocks, warps) STILL AVAILABLE on each SM
};

struct gpu_and_mem_s {
  int which_gpu; // an index into gpus_in_use
  long *mem_B;    // points to a gpu_in_use mem_B member
};

typedef struct {          //this struct will eventually be written to file. 
  int num_beacons;        //record how many tasks need scheduling.
  int num_frees;          //record how many tasks are freed. Normally, num_frees = num_beacons.
  int max_len_boomers;    //the maximum number of fetching tasks from shared queue once.
  int max_age;
  int max_observed_batch_size;
} sched_stats_t;


bemps_stopwatch_t sched_stopwatches[SCHED_NUM_STOPWATCHES];

// libstatus profiler
char LIBSTATUS_FILENAME_BASE[] = "sched";
Profiler p(LIBSTATUS_FILENAME_BASE);


//
// single-assignment scheduler
//
std::map<pid_t, int> pid_to_device_id;
std::vector<int> avail_device_ids;


//
// C:G scheduler
//
typedef struct {
    int device_id;
    int count;
} device_id_count_t;
std::map<pid_t, device_id_count_t *> pid_to_device_id_counts;
std::list<device_id_count_t *> avail_device_id_counts;
int CG_JOBS_PER_GPU = 0; // set by command-line args


//
// MGB scheduler
//
std::list<bemps_shm_comm_t *> boomers;  //store bemps_shm_comm_t that are fetched from shared queue.
std::map<pid_t, std::vector<std::pair<int, int>> > pid_to_sm_assignments; //this map store the how thread blocks and warp are assigned(to SMs)


bemps_shm_t *bemps_shm_p;
sched_stats_t stats;

sched_alg_e which_scheduler;
int max_batch_size;

int NUM_GPUS = 0; // set by init_gpus()

struct gpu_s *GPUS;
struct gpu_in_use_s *gpus_in_use;
struct gpu_and_mem_s *gpus_by_mem;

void usage_and_exit(char *prog_name) {
  printf("\n");
  printf("Usage:\n");
  printf("    %s <which_scheduler> [jobs_per_gpu]\n", prog_name);
  printf("\n");
  printf(
      "    which_scheduler is one of:\n"
      "      "
             "zero, "
             //"round-robin, "
             //"round-robin-beacons, "
             //"vector, "
             "single-assignment, "
             "cg, "
             "mgb_basic, "
             //"mgb_simple_compute, "
             "mgb, "
             "ai-heuristic\n"
      "\n"
      "    jobs_per_gpu is required and only valid for cg; it is an int that\n"
      "    specifies the maximum number of jobs that can be run a GPU\n");
  printf("\n");
  printf("\n");
  exit(1);
}

static inline long long get_time_ns(void) {
  struct timespec ts = {0};
  if (clock_gettime(CLOCK_MONOTONIC, &ts)) {
    fprintf(stderr, "ERROR: get_time_ns failed\n");
    return 0;
  }
  return (((long long)ts.tv_sec * 1000000000L) + (long long)(ts.tv_nsec));
}

static inline void dump_gpu_res(void) {
  int i;
  cudaDeviceProp prop;
  BEMPS_SCHED_LOG("Running with the following GPU resources:\n");
  for (i = 0; i < NUM_GPUS; i++) {
    cudaGetDeviceProperties(&prop, i);
    BEMPS_SCHED_LOG("  GPU " << i << " (" << prop.name << ")\n");
    BEMPS_SCHED_LOG("  mem_B: " << GPUS[i].mem_B << "\n");
    //BEMPS_SCHED_LOG("  cores: " << GPUS[i].cores << "\n");
    BEMPS_SCHED_LOG("  num_sms: " << GPUS[i].num_sms << "\n");
    BEMPS_SCHED_LOG("  thread_blocks_per_sm: " << GPUS[i].thread_blocks_per_sm << "\n");
    BEMPS_SCHED_LOG("  warps_per_sm: " << GPUS[i].warps_per_sm << "\n");
    BEMPS_SCHED_LOG("  total_thread_blocks: " << GPUS[i].total_thread_blocks << "\n");
    BEMPS_SCHED_LOG("  total_warps: " << GPUS[i].total_warps << "\n");
  }
}

static inline void init_gpus(void) {
  int i;
  cudaDeviceProp prop;

  cudaGetDeviceCount(&NUM_GPUS);
  assert(NUM_GPUS > 0 && "Must have at least 1 GPU to use the scheduler\n");

  GPUS = (struct gpu_s *) malloc(sizeof(struct gpu_s) * NUM_GPUS);
  gpus_in_use = (struct gpu_in_use_s *) malloc(sizeof(struct gpu_in_use_s) * NUM_GPUS);
  gpus_by_mem = (struct gpu_and_mem_s *) malloc(sizeof(struct gpu_and_mem_s) * NUM_GPUS);

  for (i = 0; i < NUM_GPUS; i++) {
    cudaGetDeviceProperties(&prop, i);
    GPUS[i].mem_B = prop.totalGlobalMem;
    if(strncmp(prop.name, "GeForce GTX 1080", 16) == 0){
        BEMPS_SCHED_LOG("  adjusting mem_B for GTX 1080" << "\n");
        GPUS[i].mem_B = GTX_1080_SPECS_MEM_B;
    } else if(strncmp(prop.name,"NVIDIA GeForce RTX 3080 Ti",26)==0){
        BEMPS_SCHED_LOG("  adjusting mem_B for NVIDIA GeForce RTX 3080Ti" << "\n");
        GPUS[i].mem_B = RTX_3080Ti_SEPCS_MEM_B;
    } else if(strncmp(prop.name,"NVIDIA GeForce RTX 3060",23)==0){
        BEMPS_SCHED_LOG(" adjusting mem_B for NVIDIA GeForce RTX 3060" << '\n'); 
        GPUS[i].mem_B=  RTX_3060_SPECS_MEM_B;
    }else if(strncmp(prop.name, "Tesla V100-SXM2-16GB", 20) == 0){
        BEMPS_SCHED_LOG("  adjusting mem_B for Tesla V100-SXM2-16GB" << "\n");
        GPUS[i].mem_B = V100_SXM2_SPECS_MEM_B;
    } else {
        assert(0 && "Unsupported GPU. Needs proper padding for available mem_B to work properly. Check prop.name for this device (libstatus reports in stdout and should show up in sched-log. It also gets dumped by BEMPS_SCHED_LOG if BEMPS_SCHED_DEBUG is enabled.");
    }
    GPUS[i].cores = 0; // unused
    GPUS[i].num_sms = prop.multiProcessorCount;
    GPUS[i].thread_blocks_per_sm = THREAD_BLOCKS_PER_SM;
    GPUS[i].warps_per_sm = WARPS_PER_SM;
    GPUS[i].total_thread_blocks = prop.multiProcessorCount * THREAD_BLOCKS_PER_SM;
    GPUS[i].total_warps = prop.multiProcessorCount * WARPS_PER_SM;

    gpus_in_use[i].sms = new std::vector<std::pair<int, int>>;
  }

  dump_gpu_res();
}

static inline void set_wakeup_time_ns(struct timespec *ts_p) {
  struct timespec now;

  // Must use CLOCK_REALTIME when passing to pthread_cond_timedwait
  if (clock_gettime(CLOCK_REALTIME, &now)) {
    fprintf(stderr, "ERROR: set_wakeup_time_ns failed\n");
    return;
  }

  // won't overflow
  //BEMPS_SCHED_LOG("BEMP_SCHED_TIMEOUT_NS: " << BEMPS_SCHED_TIMEOUT_NS << "\n");
  ts_p->tv_nsec = now.tv_nsec + BEMPS_SCHED_TIMEOUT_NS;
  ts_p->tv_sec = now.tv_sec + ts_p->tv_nsec / 1000000000UL;
  ts_p->tv_nsec = ts_p->tv_nsec % 1000000000UL;

  //BEMPS_SCHED_LOG("now s:  " << now.tv_sec << "\n");
  //BEMPS_SCHED_LOG("ts_p s: " << ts_p->tv_sec << "\n");
}


void dump_stats(void) {
#define STATS_LOG(str)    \
  do {                    \
    stats_file << str;    \
    stats_file.flush();   \
    BEMPS_SCHED_LOG(str); \
  } while (0)
  std::ofstream stats_file;
  bemps_stopwatch_t *sa;
  bemps_stopwatch_t *acs;
  bemps_stopwatch_t *acf;

  stats_file.open("sched-stats.out");
  sa  = &sched_stopwatches[SCHED_STOPWATCH_AWAKE];
  acs = &sched_stopwatches[SCHED_STOPWATCH_ALLOCATE_COMPUTE_SUCCESS];
  acf = &sched_stopwatches[SCHED_STOPWATCH_ALLOCATE_COMPUTE_FAIL];

  BEMPS_SCHED_LOG("Caught interrupt. Exiting.\n");
  STATS_LOG("num_beacons: " << stats.num_beacons << "\n");
  STATS_LOG("num_frees: " << stats.num_frees << "\n");
  STATS_LOG("max_len_boomers: " << stats.max_len_boomers << "\n");
  STATS_LOG("max_age: " << stats.max_age << "\n");
  STATS_LOG("max_batch_size: " << max_batch_size << "\n");
  STATS_LOG("max_observed_batch_size: "<<stats.max_observed_batch_size<< "\n");
// clockwatch is only active if BEMPS_STATS is defined
#ifdef BEMPS_STATS
  STATS_LOG("count-of-awake-times: " << sa->n << "\n");
  STATS_LOG("min-awake-time-(ns): " << sa->min << "\n");
  STATS_LOG("max-awake-time-(ns): " << sa->max << "\n");
  STATS_LOG("avg-awake-time-(ns): " << sa->avg << "\n");
  STATS_LOG("count-of-allocate-compute-success-times: " << acs->n << "\n");
  STATS_LOG("min-acs-time-(ns): " << acs->min << "\n");
  STATS_LOG("max-acs-time-(ns): " << acs->max << "\n");
  STATS_LOG("avg-acs-time-(ns): " << acs->avg << "\n");
  STATS_LOG("count-of-allocate-compute-fail-times: " << acf->n << "\n");
  STATS_LOG("min-acf-time-(ns): " << acf->min << "\n");
  STATS_LOG("max-acf-time-(ns): " << acf->max << "\n");
  STATS_LOG("avg-acf-time-(ns): " << acf->avg << "\n");
#endif

  stats_file.close();
}

void sigint_handler(int unused) {
  BEMPS_SCHED_LOG("Caught interrupt. Exiting.\n");
  p.stop_sampling();
  dump_stats();
  exit(0);
}

struct MemFootprintCompare {
  bool operator()(const bemps_shm_comm_t *lhs, const bemps_shm_comm_t *rhs) {
    // sort in increasing order. priority queue .top() and pop() will
    // therefore return the largest values
    // return lhs->beacon.mem_B < rhs->beacon.mem_B;

    // sort in decreasing order. iterating use vector begin() and end().
    // return lhs->beacon.mem_B > rhs->beacon.mem_B;

    // sort in decreasing order. iterate with index.
    return lhs->beacon.mem_B > rhs->beacon.mem_B;

    // sort in increasing order. iterate with index. use back() and pop_back()
    // return lhs->beacon.mem_B < rhs->beacon.mem_B;
  }
} mem_footprint_compare;

struct AvailDevicesCompare {
  bool operator()(const device_id_count_t *lhs, const device_id_count_t *rhs) {
    // sort in decreasing order. iterate with index.
    return lhs->count > rhs->count;
  }
} avail_devices_compare;


int sort_gpu_by_mem_in_use(const void *a_arg, const void *b_arg) {
  struct gpu_and_mem_s *a = (struct gpu_and_mem_s *) a_arg;
  struct gpu_and_mem_s *b = (struct gpu_and_mem_s *) b_arg;
  if (*a->mem_B < *b->mem_B) {
    return -1;
  }
  return 1;
}


// Emulate the hardware's round-robin allocation to find the next available
// streaming multiprocessor
int get_next_avail_sm(std::vector<std::pair<int, int> > &sms,
                      std::vector<std::pair<int, int> > &rq,
                      int num_warps,
                      int curr_sm) {
  int num_sms;
  int sm_idx;

  if (sms[curr_sm].first && sms[curr_sm].second >= num_warps) {
    //BEMPS_SCHED_LOG("Quick availability curr_sm: " << curr_sm << "\n");
    return curr_sm;
  }

  num_sms = sms.size();
  sm_idx  = (curr_sm + 1) % num_sms;

  while (sm_idx != curr_sm) {
    if ((sms[sm_idx].first - rq[sm_idx].first) &&
      (sms[sm_idx].second - rq[sm_idx].second) >= num_warps) {
      //BEMPS_SCHED_LOG("Found an available sm_idx: " << sm_idx << "\n");
      return sm_idx;
    }
    sm_idx = (sm_idx + 1) % num_sms;
  }

  //BEMPS_SCHED_LOG("No available SM. returning -1\n");
  return -1;
}


bool saturates_compute(struct gpu_s *GPU,
                       bemps_shm_comm_t *comm) {
  if (GPU->total_warps < comm->beacon.warps) {
    BEMPS_SCHED_LOG("total warps (" << GPU->total_warps << ") "
                    << "less than beacon warps (" << comm->beacon.warps << "). "
                    << "Returning true (saturates compute)\n");
    return true;
  }
  if (GPU->total_thread_blocks < comm->beacon.thread_blocks) {
    BEMPS_SCHED_LOG("total thread blocks (" << GPU->total_thread_blocks << ") "
                    << "less than beacon thread blocks ("
                    << comm->beacon.thread_blocks << "). "
                    << "Returning true (saturates compute)\n");
    return true;
  }
  BEMPS_SCHED_LOG("Does not saturate compute. returning false\n");
  return false;
}


// emulate SM allocation of a GPU device.
bool allocate_compute(struct gpu_s *GPU,
                      struct gpu_in_use_s *gpu_in_use,
                      bemps_shm_comm_t *comm) {
  std::vector<std::pair<int,int>> &sms = *gpu_in_use->sms;
  int num_thread_blocks = comm->beacon.thread_blocks;
  // ignore off-by-one
  int num_warps_per_block = comm->beacon.warps / comm->beacon.thread_blocks;
  int avail_sm;
  int i;
  int tmp_curr_sm;

  bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_ALLOCATE_COMPUTE_SUCCESS]);
  bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_ALLOCATE_COMPUTE_FAIL]);


  // If the GPU has no jobs running on it, and this job would saturate all
  // the compute units, then don't both with SM assignment. Just mark it as
  // compute-saturated and return true (to say that the GPU should be allocated)
  if (saturates_compute(GPU, comm) && gpu_in_use->active_jobs == 0){  //quick allocation
    BEMPS_SCHED_LOG("Compute would be saturated and there are no active jobs.\n");
    gpu_in_use->compute_saturated = 1;
    return true;
  }

  // If the GPU is already compute-saturated, then return false (to say that
  // we can't allocate anything else on it)
  if (gpu_in_use->compute_saturated) {
    BEMPS_SCHED_LOG("Compute is already saturated\n");
    assert(gpu_in_use->active_jobs == 1);
    return false;
  }

  // A vector of length num_sms. Each element is a pair:
  //   occupied thread blocks for that SM
  //   occupied warps for that SM
  std::vector<std::pair<int, int>> rq(GPU->num_sms, {0, 0});

  //BEMPS_SCHED_LOG("num_thread_blocks: " << num_thread_blocks << "\n");
  tmp_curr_sm = gpu_in_use->curr_sm;
  while (num_thread_blocks) {
    avail_sm = get_next_avail_sm(sms, rq, num_warps_per_block, tmp_curr_sm);
    if (avail_sm < 0) {
      bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_ALLOCATE_COMPUTE_FAIL]);
      BEMPS_SCHED_LOG("Compute not available for pid: " << comm->pid << "\n");
      return false;
    }
    rq[avail_sm].first  += 1;
    rq[avail_sm].second += num_warps_per_block;
    tmp_curr_sm = (avail_sm + 1) % GPU->num_sms;
    num_thread_blocks--;
  }

  // ... all thread blocks have been assigned to an SM.
  // commit the assignments, and update the SMs
  BEMPS_SCHED_LOG("Committing compute resources for pid: " << comm->pid << "\n");
  gpu_in_use->curr_sm = tmp_curr_sm;
  pid_to_sm_assignments[comm->pid] = rq;
  for (i = 0; i < GPU->num_sms; i++) {
    sms[i].first  -= rq[i].first;
    sms[i].second -= rq[i].second;
  }

  bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_ALLOCATE_COMPUTE_SUCCESS]);
  return true;
}


void release_compute(struct gpu_s *GPU,
                     struct gpu_in_use_s *gpu_in_use,
                     bemps_shm_comm_t *comm) {
  int i;
  std::vector<std::pair<int,int>> &sms = *gpu_in_use->sms;  //gpu_in_use->sms records the number of thread blocks and warps that are allocated to any one of SMs in the GPU.

  if (gpu_in_use->compute_saturated) {
    gpu_in_use->compute_saturated = 0;
    return;
  }

  BEMPS_SCHED_LOG("Freeing compute resources for pid: " << comm->pid << "\n");
  std::vector<std::pair<int,int>> &s = pid_to_sm_assignments[comm->pid];
  for (i = 0; i < GPU->num_sms; i++) {
    sms[i].first  += s[i].first;
    sms[i].second += s[i].second;
  }
  pid_to_sm_assignments.erase(comm->pid);
}


//This function wraps up those integer linear algorithm,
//such that we can choose which specifc algorithm to address our integer linear probelm.
//solve_alg_e used to sepcify which integer linear algorithm to use.
//If the return object, std::list::size()==0, that indicates solver doesn't find a feasible or optimal solution.
//This should never modify (push or pop) the unscheduled_list. Because the solution from here is not necessarily
//the expected solution. The expected solution makes the epsilon minimum.
std::list<bemps_shm_comm_t*> integer_linear_solver(std::list<bemps_shm_comm_t*>& unscheduled_list,int64_t epsilon,
                                                  float ai_ridge,solve_alg_e SOLVE_ALG_TYPE=solve_alg_e::SOLVE_ALG_ZERO_E)
{
  BEMPS_SCHED_LOG("Unscheduled list size: " << unscheduler_list.size() << '\n');
  BEMPS_SCHED_LOG("epsilon: " << epsilon <<'\n');
  BEMPS_SCHED_LOG("ai_ridge: " << ai_ridge <<'\n');
  BEMPS_SCHED_LOG("algorithm type: " << SOLVE_ALG_TYPE << '\n');

  std::list<bemps_shm_comm_t*> return_list;

  if(SOLVE_ALG_TYPE==solve_alg_e::SOLVE_ALG_ZERO_E){
    operations_research::sat::CpModelBuilder cp_model;
    const operations_research::Domain domain(0,1);
    std::vector<operations_research::sat::IntVar> var_vec;
    int unscheduled_list_size=unscheduled_list.size();

    operations_research::sat::LinearExpr accm_x_times_F;
    operations_research::sat::LinearExpr accm_x_times_B;
    operations_research::sat::LinearExpr accm_x_times_M;

    operations_research::sat::LinearExpr r_accm_x_times_B;
    operations_research::sat::LinearExpr epsilon_accm_x_times_B;
    operations_research::sat::LinearExpr minus_epsilon_accm_x_times_B;
    //declare model variable xi 
    for(int i=0;i<unscheduled_list_size;i++)
      var_vec.push_back(cp_model.NewIntVar(domain));
  
    // constrcut linear expression
    int var_index=0;  //remember that set to zero when used to index
    for(std::list<bemps_shm_comm_t*>::iterator b_itera=unscheduled_list.begin(),
                                               e_itera=unscheduled_list.end();
                                               b_itera!=e_itera;
                                               b_itera++)
    {
      int64_t F_i = static_cast<int64_t>((*b_itera)->beacon.num_fp);
      int64_t B_i = static_cast<int64_t>((*b_itera)->beacon.num_tb);
      int64_t M_i = static_cast<int64_t>((*b_itera)->beacon.mem_B);

      operations_research::sat::LinearExpr tmp_xF=
      operations_research::sat::LinearExpr(var_vec[var_index])*F_i;
      operations_research::sat::LinearExpr tmp_xB=
      operations_research::sat::LinearExpr(var_vec[var_index])*B_i;
      operations_research::sat::LinearExpr tmp_xM=
      operations_research::sat::LinearExpr(var_vec[var_index])*M_i;

      accm_x_times_F+=tmp_xF;
      accm_x_times_B+=tmp_xB; 

      var_index+=1;
    }

    r_accm_x_times_B=accm_x_times_B*static_cast<int64_t>(ai_ridge);
    epsilon_accm_x_times_B=accm_x_times_B*epsilon;
    minus_epsilon_accm_x_times_B=epsilon_accm_x_times_B*(-1);

    //add model constraint
    cp_model.AddLessOrEqual(minus_epsilon_accm_x_times_B,
                            accm_x_times_F-r_accm_x_times_B);
    cp_model.AddLessOrEqual(accm_x_times_F-r_accm_x_times_B,
                            epsilon_accm_x_times_B);
    
    //Solving problem
    const operations_research::sat::CpSolverResponse response = 
    operations_research::sat::Solve(cp_model.Build());

    if(response.status()==operations_research::sat::CpSolverStatus::OPTIMAL ||
       response.status()==operations_research::sat::CpSolverStatus::FEASIBLE ){
        var_index=0;
        for(std::list<bemps_shm_comm_t*>::iterator b_itera=unscheduled_list.begin(),
                                                   e_itera=unscheduled_list.end();
                                                   b_itera!=e_itera;
                                                   b_itera++)
        {
          if(operations_research::sat::SolutionIntegerValue(response,var_vec[var_index])==1)
            return_list.push_back(*b_itera);
          ++var_index;
        }
    }else{
      BEMPS_SCHED_LOG("Not find a feasible or optimal solution\n");
    }

  }
  return return_list;
}

//This function uses binary search to find the solution such that the inequality satifies the minimum epsilon.
//The epsilon ranges from [0,output_value]. We leave how to set a best maximum epsilon such that it can speed up the algorithm behind.
//Up to now, we first try the value as int64_t 100.
std::list<bemps_shm_comm_t*> binary_search_to_find_solution(std::list<bemps_shm_comm_t*>& unscheduled_list, 
                                                          float ai_ridge, int64_t max_epsilon=100, 
                                                          solve_alg_e SOLVE_ALG_TYPE=solve_alg_e::SOLVE_ALG_ZERO_E)
{
  int64_t left=0;
  int64_t right=max_epsilon;
  int64_t middle_epsilon;	//we hope target_epsilon as small as possible
	std::list<bemps_shm_comm_t*> return_list;
  assert(!unscheduled_list.empty()&&"Unscheduled list must not be empty\n");

    while(left<=right){
        std::list<bemps_shm_comm_t*> tmp_return_list;
        int64_t middle_epsion=(left+right)/2;
        //If no solution if found, this function return emtpy.
    	  tmp_return_list=integer_linear_solver(unscheduled_list,middle_epsilon,
                	                          ai_ridge, SOLVE_ALG_TYPE);
        if(left==right){
        	if(!tmp_return_list.empty())
                return_list=tmp_return_list;
            break;
        }else{
            if(!tmp_return_list.empty())
                right=middle_epsilon-1;
            else
                left=middle_epsilon+1;
        }
    }
    
    assert(!return_list.empty()&&"Solver can't find a feasible solution,which is very wired in our custom algorithm. So terminate the program now\n");
    
    //After breaking the above loop, those bemps_shm_comm_t to be scheduled have been determined. They should be removed from unscheduled_list.
    //FIXME: Is there a more efficient algorithm to remove those kernel to be scheduled?
    for(std::list<bemps_shm_comm_t*>::iterator b_sched_itera=return_list.begin(),
       										   e_sched_itera=return_list.end();
       										   b_sched_itera!=e_sched_itera;
       										   b_sched_itera++)
    {
       for(std::list<bemps_shm_comm_t*>::iterator b_unsched_itera=unscheduled_list.begin(),
          										  e_unsched_itera=unscheduled_list.end();
          										  b_unsched_itera!=e_unsched_itera;
          										  b_unsched_itera++)
       {
           if(*b_unsched_itera==*b_sched_itera)
               //erase() rather than remove. erase() removes element by its position (iteration)
               unscheduled_list.erase(b_unsched_itera);
       }
    }
    
    return return_list;
}

//Our heuristic algorithm is designed to fit different version GPU, which have different ridge arithmetic intensity
//Assuming the GPU-system with single or multiple homogeneous GPUs, the parameter ai_ridge represents ridge arithmetic intensity(>0)
void sched_ai_heuristic(float ai_ridge){ //heuristic scheduling algorithm based on kernel's arithmetic intensity
  int tmp_dev_id;
  int* head_p;
  int* tail_p;
  int* jobs_running_on_gpu;
  int* jobs_waiting_on_gpu;
  int assigned;
  struct timespec ts;
  int boomers_len;
  int i;
  int g;
  bemps_shm_comm_t* comm;
  int batch_size;
  int which_gpu;
  bool allocated;
  //long mem_max; //not need, gurantee memory safe when decision-making.
  //long mem_in_use;
  //long mem_to_add;

  /*
    Here it should be a queue rather than a list.
    Because it's supposed to first in, firt out.
  */ 
  std::queue<std::list<bemps_shm_comm_t*>> ready_queue;

  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;
  jobs_running_on_gpu = &bemps_shm_p->gen->jobs_running_on_gpu;
  jobs_waiting_on_gpu = &bemps_shm_p->gen->jobs_waiting_on_gpu;


  assert((ai_ridge>=0)&&"Invaild ridge arithmetic intensity\n");

  while(1){
    set_wakeup_time_ns(&ts);
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    pthread_cond_timedwait(&bemps_shm_p->gen->cond,&bemps_shm_p->gen->lock,
                            &ts);
    bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]); 
    ALIVE_MSG(); 
    
    batch_size=0;
    // First loop: This while loop is used to fetch the beacon to be scheduled in shared memory
    // In this loop, compared to other mgb algorithm ,we don't track thread blocks allocation
    while(*tail_p!=*head_p){
      BEMPS_SCHED_LOG("*head_p: "<<(*head_p)<<'\n');
      BEMPS_SCHED_LOG("*tail_p: "<<(*tail_p)<<'\n');

      comm = &bemps_shm_p->comm[*tail_p];

      while(comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E){
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if(comm->exit_flag){
        BEMPS_SCHED_LOG("seening exit flag\n");
        comm->exit_flag=0; 
      } else{
        assert(comm->beacon.mem_B);
        BEMPS_SCHED_LOG("First loop seening mem_B: "<<comm->beacon.mem_B 
                                                    <<"\n");
        if(comm->beacon.mem_B<0){
          BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
          stats.num_frees++;
          tmp_dev_id = comm->sched_notif.device_id;
          //Add (don't substract), because mem_B is negative already
          long tmp_bytes_to_free = comm->beacon.mem_B;  
          long tmp_warps_to_free = comm->beacon.warps;
          BEMPS_SCHED_LOG("Freeing "<<tmp_bytes_to_free << "bytes "
                                    <<"from device" << tmp_dev_id <<"\n");
          BEMPS_SCHED_LOG("Freeing "<<tmp_warps_to_free << " warps "
                                   <<"from device "<< tmp_warps_to_free<< "\n" );
          gpus_in_use[tmp_dev_id].mem_B += tmp_bytes_to_free;
          gpus_in_use[tmp_dev_id].warps += tmp_warps_to_free;

          //in this custom algorithm, we don't track the the allocation of mapping thread block to SMs
          //release_compute(&GPUS[tmp_dev_id],&gpus_in_use[tmp_dev_id],comm)

          gpus_in_use[tmp_dev_id].active_jobs--;
          /*We use gpus_in_use[index].compute_saturate to indicate
          whether the whole concurrent set is completed.
          compute_saturated=0 means we are able to issue concurrent set to this GPU*/
          if(gpus_in_use[tmp_dev_id].active_jobs==0)
            gpus_in_use[tmp_dev_id].compute_saturated=0;
          --*jobs_running_on_gpu;
        } else {
          stats.num_beacons++;
          boomers.push_back(comm);
          batch_size++;
          ++*jobs_waiting_on_gpu;
        }
      }
      
      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
  
    if(batch_size > stats.max_observed_batch_size) {
      stats.max_observed_batch_size = batch_size;
    }

    boomers_len = boomers.size();
    if(boomers_len>stats.max_len_boomers){
      stats.max_len_boomers = boomers_len;
    }
    if(boomers_len>0){
      BEMPS_SCHED_LOG("boomers_len: " << boomers_len <<"\n");
    }

    //Second loop: use Google or-tools to schedule kernel according to their arithmetic intensity
    //The problem we need to address is a integer linear prgramming problem which is a NP-hard problem
    //in this case, we use Google or-tools to handle it.
    //In this algorithm, we premise the following assmptions:
    //  -Every kernel has more thread blocks than device's SMs. 
    //    Based on this assumption and the assignment algorithm of thread blocks to SM, which is round-robin,
    //    we don't have to find another kernel when the collective arithmetic intensity of concurrent set 
    //    achieves ridge point of the device.

    /*Not sort beacon by their memory footprint, 
      because we don't schedule beacon by taking considering of
      their memory footprint.
    */

    //boomers.sort(mem_footprint_compare);

    /*
    Decision process. We should measure the period of time it takes.
    */
    if(!boomers.empty()){
      bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_DECISION_MAKING]);
      std::list<bemps_shm_comm_t*> return_list = 
      binary_search_to_find_solution(boomers,ai_ridge);
      assert(!return_list.empty()\
      && "Not get a feasible concurrent set, which is impossible if boomers not empty\n");
      bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_DECISION_MAKING]);
      ready_queue.push(return_list);
    }

    BEMPS_SCHED_LOG("ready_queue len: " << ready_queue.size() <<"\n");

    assigned=0;
    std::list<bemps_shm_comm_t*> one_off;
    for(g=0; g<NUM_GPUS; ++g){
      if(ready_queue.size()==0) //without this statement, std::queue::fornt() will cause segment fault once ready_queue is emtpy.
        break;
    do{
      one_off = ready_queue.front(); //visit but not pop it right now;
      int one_off_size = one_off.size(); 

      /* The process of observd max age of eac beacon  can be put together with modifying age
      for(auto b_itera=one_off.begin(),e_itera=one_off.end();
          b_itera!=e_itera; ++b_itera){
        if((*b_itera)->age > stats.max_age){
          stats.max_age = (*b_itera)->age;
        }
      }
      */
      /*
      In mgb or mgb_basic algorithm, for every GPU, we need to check 
      whether the GPU has sufficient memory and sufficient compute resource.
      However, in this algorithm, we only check whether the GPU is compute-saturated
      (or whether there is no any active job on the device.)
      */
      if(gpus_in_use[g].compute_saturated==0){
        long tmp_bytes_to_allocate = 0;
        long tmp_warps_to_allocate = 0;

        for(auto b_itera=one_off.begin(),e_itera=one_off.end();
        b_itera!=e_itera; ++b_itera){
              tmp_bytes_to_allocate+=(*b_itera)->beacon.mem_B;
              tmp_warps_to_allocate+=(*b_itera)->beacon.warps;
              (*b_itera)->sched_notif.device_id = g;          //set target GPU for each beacon in the concurrent set
              (*b_itera)->state = BEMPS_BEACON_STATE_BEACON_FIRED_E;
        }

        gpus_in_use[g].active_jobs=one_off_size;
        gpus_in_use[g].mem_B=tmp_bytes_to_allocate;
        gpus_in_use[g].warps=tmp_warps_to_allocate;
        gpus_in_use[g].compute_saturated=1;
        jobs_waiting_on_gpu-=one_off_size;
        jobs_running_on_gpu+=one_off_size;
        assigned=1;
        ready_queue.pop();
        break;
      }
      g++;
    }while(g < NUM_GPUS);

    if(!assigned){
      //modifyng the observed max and beacon's age
      for(auto b_itera=one_off.begin(),e_itera=one_off.end();
          b_itera!=e_itera; ++b_itera){
            if((*b_itera)->age > stats.max_age)
              stats.max_age = (*b_itera)->age;
            (*b_itera)->age++;
      }
    }else{
      for(auto b_itera=one_off.begin(),e_itera=one_off.end();
          b_itera!=e_itera ; ++b_itera){
            sem_post(&((*b_itera)->sched_notif.sem));
      }
    }
    }
    bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
  }
}


// Our custom scheduler, multi-GPU with beacons
// This algorithm does the following:
//   - Finds the GPU with the most available memory and which can also fit
//     the kernel
//       -- this becomes a sort. Previously it was not.
//   - Checks that it can support the compute requirements, as well.
//   - If it can, it puts the kernel there.
//   - If it can't, it checks the GPU with next most available memory.
// There are other ways to do this, e.g.
// Instead of sorting the boomers based on memory, use some vector weighting
// technique (b/c now we have warps and thread blocks).
// Or instead of finding the GPU with the most available memory, use
// round-robin, and just check that memory, warps, and thread blocks fit.
// Or instead of stopping after finding the first GPU (with most available
// memory) that can support the compute requirements, exhaust all of them to
// find the one that also has the least compute load.
void sched_mgb(void) {//Alg.2  //account for whether both requirement for thread blcoks and warps are met.
  int tmp_dev_id;
  int *head_p;
  int *tail_p;
  int *jobs_running_on_gpu;
  int *jobs_waiting_on_gpu;
  int assigned;
  struct timespec ts;
  int boomers_len;
  int i;
  int g;
  bemps_shm_comm_t *comm;
  int batch_size;
  int which_gpu;
  bool allocated;
  long mem_max;
  long mem_in_use;
  long mem_to_add;

  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;
  jobs_running_on_gpu = &bemps_shm_p->gen->jobs_running_on_gpu;
  jobs_waiting_on_gpu = &bemps_shm_p->gen->jobs_waiting_on_gpu;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
                           &ts);
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);
    bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);

    ALIVE_MSG();

    // First loop: Catch the scheduler's tail back up with the beacon
    // queue's head. If we see a free-beacon, then reclaim that resource.
    //BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
    //BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
    batch_size = 0;
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if (comm->exit_flag) {
        BEMPS_SCHED_LOG("seeing exit flag\n");
        comm->exit_flag = 0;
      } else {
        assert(comm->beacon.mem_B);
        BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
                                                    << "\n");
        if (comm->beacon.mem_B < 0) { //kernel finished, release resource
          BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
          stats.num_frees++;
          tmp_dev_id = comm->sched_notif.device_id;
          // Add (don't subtract), because mem_B is negative already
          long tmp_bytes_to_free = comm->beacon.mem_B;
          long tmp_warps_to_free = comm->beacon.warps;
          BEMPS_SCHED_LOG("Freeing " << tmp_bytes_to_free << " bytes "
                          << "from device " << tmp_dev_id << "\n");
          BEMPS_SCHED_LOG("Freeing " << tmp_warps_to_free << " warps "
                          << "from device " << tmp_dev_id << "\n");
          gpus_in_use[tmp_dev_id].mem_B += tmp_bytes_to_free;
          gpus_in_use[tmp_dev_id].warps += tmp_warps_to_free;
          release_compute(&GPUS[tmp_dev_id], &gpus_in_use[tmp_dev_id], comm);
          gpus_in_use[tmp_dev_id].active_jobs--;
          --*jobs_running_on_gpu;
        } else {                    //kernel not issued yet, set it waiting 
          stats.num_beacons++;
          boomers.push_back(comm);
          batch_size++; // batch size doesn't include free() beacons
          ++*jobs_waiting_on_gpu;
        }
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }

    if (batch_size > stats.max_observed_batch_size) {
      stats.max_observed_batch_size = batch_size;
    }

    // Second loop: Walk the boomers. This time handle regular beacons, and
    // attempt to assign them to a device. The boomers are sorted by memory
    // footprint, highest to lowest.
    boomers.sort(mem_footprint_compare);
    boomers_len = boomers.size();
    if (boomers_len > stats.max_len_boomers) {
      stats.max_len_boomers = boomers_len;
    }
    if (boomers_len > 0) {
      BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
    }
    for (i = 0; i < boomers_len; i++) {
      assigned = 0;
      comm = boomers.front();
      boomers.pop_front();

      if (comm->age > stats.max_age) {
        stats.max_age = comm->age;
      }

      // The target device for a process must have memory available for it,
      // and it must also have sufficient space on its SMs to support its
      // thread block and warp requirements.
      int target_dev_id = 0;

      // sort the GPUs based by memory that's in use, from least to greatest
      qsort(gpus_by_mem, NUM_GPUS, sizeof(struct gpu_and_mem_s), sort_gpu_by_mem_in_use);   //struct gpu_and_mem_s{
                                                                                                //int which_gpu;
      // Now attempt to assign the process to a GPU.                                            //long* mem_B;
      g = 0;                                                                                //}
      do {
        which_gpu  = gpus_by_mem[g].which_gpu;
        assert(*gpus_by_mem[g].mem_B == gpus_in_use[which_gpu].mem_B);
        mem_max    = GPUS[which_gpu].mem_B;
        mem_in_use = gpus_in_use[which_gpu].mem_B;
        mem_to_add = comm->beacon.mem_B;
        BEMPS_SCHED_LOG("mem_max: "    << mem_max << "\n");
        BEMPS_SCHED_LOG("mem_in_use: " << mem_in_use << "\n");
        BEMPS_SCHED_LOG("mem_to_add: " << mem_to_add << "\n");

        if ((mem_in_use + mem_to_add) > mem_max) {
          BEMPS_SCHED_LOG("No space available\n");
          break;
        }

        allocated = allocate_compute(&GPUS[which_gpu],
                                     &gpus_in_use[which_gpu],
                                     comm);
        if (allocated) {  //if allocation fails , try the next GPU
          target_dev_id = which_gpu;
          assigned = 1;
          break;
        }
        g++;
      } while (g < NUM_GPUS);

      if (!assigned) {
        // FIXME: need to add stats, and possibly a way to reserve a
        // GPU to prevent starving.
        comm->age++;
        boomers.push_back(comm);
        // don't adjust jobs-waiting-on-gpu. it was incremented when job first
        // went into the boomers list
      } else {
        // XXX For this algorithm, the allocate_compute() function relies on the
        // "sms" vector and not the totals for warps.
        // Nevertheless, we track it here for completeness.
        long tmp_bytes_to_add = comm->beacon.mem_B;
        long tmp_warps_to_add = comm->beacon.warps;
        BEMPS_SCHED_LOG("Adding " << tmp_bytes_to_add << " bytes "
                        << "to device " << target_dev_id << "\n");
        BEMPS_SCHED_LOG("Adding " << tmp_warps_to_add << " warps "
                        << "to device " << target_dev_id << "\n");
        gpus_in_use[target_dev_id].mem_B += tmp_bytes_to_add;
        gpus_in_use[target_dev_id].warps += tmp_warps_to_add;
        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
                                            << "on device(" << target_dev_id
                                            << ")\n");
        // FIXME Is this SCHEDULER_READ state helping at all?
        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = target_dev_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
        gpus_in_use[target_dev_id].active_jobs++;
        ++*jobs_running_on_gpu;
        --*jobs_waiting_on_gpu;
      }
    }
    bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
  }
}



// Our custom scheduler, multi-GPU with beacons.
// This algorithm adds a "simple compute" part to the original ("basic")
// algorithm. Previously the GPU with available memory and the least number of
// active warps was chosen. Here, we also check that the warps and thread
// blocks are beneath some SCHED_MGB_SIMPLE_COMPUTE_THRESHOLD. The idea is that perfect
// tracking of available SMs (in terms of warp and thread block requirements)
// is slow and complex; we know that the ideal assignment of kernels to
// SMs will saturate 100% maximum warps and 100% maximum thread blocks; but
// this perfect saturation is unlikely, so we assume that the compute units
// are saturated once the SCHED_MGB_SIMPLE_COMPUTE_THRESHOLD is reached (e.g. 80% max
// warps and thread blocks).
//
// Deprecated
//
//void sched_mgb_simple_compute(void) {
//  int tmp_dev_id;
//  int *head_p;
//  int *tail_p;
//  int *jobs_running_on_gpu;
//  int *jobs_waiting_on_gpu;
//  int assigned;
//  struct timespec ts;
//  int boomers_len;
//  int i;
//  bemps_shm_comm_t *comm;
//  int batch_size;
//
//  head_p = &bemps_shm_p->gen->beacon_q_head;
//  tail_p = &bemps_shm_p->gen->beacon_q_tail;
//  jobs_running_on_gpu = &bemps_shm_p->gen->jobs_running_on_gpu;
//  jobs_waiting_on_gpu = &bemps_shm_p->gen->jobs_waiting_on_gpu;
//
//  while (1) {
//    set_wakeup_time_ns(&ts);
//
//    // wait until we get a signal or time out
//    pthread_mutex_lock(&bemps_shm_p->gen->lock);
//    // TODO spurious wakeups ? shouldn't make a big difference to wake up
//    // randomly from time to time before the batch is ready
//    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
//                           &ts);
//    pthread_mutex_unlock(&bemps_shm_p->gen->lock);
//    bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
//
//    ALIVE_MSG();
//
//    // First loop: Catch the scheduler's tail back up with the beacon
//    // queue's head. If we see a free-beacon, then reclaim that resource.
//    //BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
//    //BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
//    batch_size = 0;
//    while (*tail_p != *head_p) {
//      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
//      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
//
//      comm = &bemps_shm_p->comm[*tail_p];
//      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
//        // TODO probably want to track a stat for this case
//        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
//                        << "was set. (Not a bug, but unless we're "
//                        << "flooded with beacons, this should be rare."
//                        << "\n");
//        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
//        // FIXME sync shouldn't hurt, but may not help?
//        __sync_synchronize();
//      }
//
//      if (comm->exit_flag) {
//        BEMPS_SCHED_LOG("seeing exit flag\n");
//        comm->exit_flag = 0;
//      } else {
//        assert(comm->beacon.mem_B);
//        BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
//                                                    << "\n");
//        if (comm->beacon.mem_B < 0) {
//          BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
//          stats.num_frees++;
//          tmp_dev_id = comm->sched_notif.device_id;
//          // Add (don't subtract), because mem_B is negative already
//          long tmp_bytes_to_free = comm->beacon.mem_B;
//          long tmp_warps_to_free = comm->beacon.warps;
//          BEMPS_SCHED_LOG("Freeing " << tmp_bytes_to_free << " bytes "
//                          << "from device " << tmp_dev_id << "\n");
//          BEMPS_SCHED_LOG("Freeing " << tmp_warps_to_free << " warps "
//                          << "from device " << tmp_dev_id << "\n");
//          gpus_in_use[tmp_dev_id].mem_B += tmp_bytes_to_free;
//          gpus_in_use[tmp_dev_id].warps += tmp_warps_to_free;
//          gpus_in_use[tmp_dev_id].active_jobs--;
//          --*jobs_running_on_gpu;
//        } else {
//          stats.num_beacons++;
//          boomers.push_back(comm);
//          batch_size++; // batch size doesn't include free() beacons
//          ++*jobs_waiting_on_gpu;
//        }
//      }
//
//      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
//    }
//
//    if (batch_size > stats.max_observed_batch_size) {
//      stats.max_observed_batch_size = batch_size;
//    }
//
//    // Second loop: Walk the boomers. This time handle regular beacons, and
//    // attempt to assign them to a device. The boomers are sorted by memory
//    // footprint, highest to lowest.
//    boomers.sort(mem_footprint_compare);
//    boomers_len = boomers.size();
//    if (boomers_len > stats.max_len_boomers) {
//      stats.max_len_boomers = boomers_len;
//    }
//    if (boomers_len > 0) {
//      BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
//    }
//    for (i = 0; i < boomers_len; i++) {
//      assigned = 0;
//      comm = boomers.front();
//      boomers.pop_front();
//
//      if (comm->age > stats.max_age) {
//        stats.max_age = comm->age;
//      }
//
//      // The target device for a process must have memory available for it,
//      // and it should be the device with the least warps currently in use.
//      long curr_min_warps = LONG_MAX;
//      int target_dev_id = 0;
//      for (tmp_dev_id = 0; tmp_dev_id < NUM_GPUS; tmp_dev_id++) {
//        BEMPS_SCHED_LOG("Checking device "           << tmp_dev_id << "\n"
//                        << "  Total avail bytes: "   << GPUS[tmp_dev_id].mem_B << "\n"
//                        << "  In-use bytes: "        << gpus_in_use[tmp_dev_id].mem_B << "\n"
//                        << "  Trying-to-fit bytes: " << comm->beacon.mem_B << "\n"
//                        << "  In-use warps: "        << gpus_in_use[tmp_dev_id].warps << "\n"
//                        << "  Trying-to-add warps: " << comm->beacon.warps << "\n");
//        if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
//             GPUS[tmp_dev_id].mem_B)) {
//          BEMPS_SCHED_LOG("fits mem\n");
//          if (gpus_in_use[tmp_dev_id].warps < curr_min_warps) {
//              curr_min_warps = gpus_in_use[tmp_dev_id].warps;
//              target_dev_id = tmp_dev_id;
//              assigned = 1;
//              BEMPS_SCHED_LOG("warps are min. target_dev_id set to " << target_dev_id << "\n");
//          }
//        } else {
//            BEMPS_SCHED_LOG("Couldn't fit " << comm->beacon.mem_B << "\n");
//        }
//      }
//
//      // This is the change for "simple compute". A little hacky, but we know
//      // that the target device is the one with the fewest actie warps. So we
//      // only have to check that one to see if it's beneath the threshold.
//      if (assigned) {
//        float max_tbs   = 1.0f * GPUS[target_dev_id].total_thread_blocks;
//        float max_warps = 1.0f * GPUS[target_dev_id].total_warps;
//        float warp_percentage =
//          1.0f
//          * (gpus_in_use[target_dev_id].warps + comm->beacon.warps)
//          / max_warps;
//        BEMPS_SCHED_LOG("max_tbs: " << max_tbs << "\n");
//        BEMPS_SCHED_LOG("max_warps: " << max_warps << "\n");
//        BEMPS_SCHED_LOG("warp_percentage: " << warp_percentage << "\n");
//        BEMPS_SCHED_LOG("active_jobs: " << gpus_in_use[target_dev_id].active_jobs << "\n");
//        if (gpus_in_use[target_dev_id].active_jobs < SCHED_MGB_SIMPLE_COMPUTE_MIN_JOBS) {
//          BEMPS_SCHED_LOG("assigned A\n");
//          assigned = 1;
//        } else if (warp_percentage < SCHED_MGB_SIMPLE_COMPUTE_THRESHOLD) {
//          BEMPS_SCHED_LOG("assigned B\n");
//          assigned = 1;
//        } else {
//          BEMPS_SCHED_LOG("unassigned\n");
//          assigned = 0;
//        }
//      }
//
//      if (!assigned) {
//        // FIXME: need to add stats, and possibly a way to reserve a
//        // GPU to prevent starving.
//        comm->age++;
//        boomers.push_back(comm);
//        // don't adjust jobs-waiting-on-gpu. it was incremented when job first
//        // went into the boomers list
//      } else {
//        long tmp_bytes_to_add = comm->beacon.mem_B;
//        long tmp_warps_to_add = comm->beacon.warps;
//        BEMPS_SCHED_LOG("Adding " << tmp_bytes_to_add << " bytes "
//                        << "to device " << target_dev_id << "\n");
//        BEMPS_SCHED_LOG("Adding " << tmp_warps_to_add << " warps "
//                        << "to device " << target_dev_id << "\n");
//        gpus_in_use[target_dev_id].mem_B += tmp_bytes_to_add;
//        gpus_in_use[target_dev_id].warps += tmp_warps_to_add;
//        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
//                                            << "on device(" << target_dev_id
//                                            << ")\n");
//        // FIXME Is this SCHEDULER_READ state helping at all?
//        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
//        comm->sched_notif.device_id = target_dev_id;
//        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
//        sem_post(&comm->sched_notif.sem);
//        gpus_in_use[target_dev_id].active_jobs++;
//        ++*jobs_running_on_gpu;
//        --*jobs_waiting_on_gpu;
//      }
//    }
//    bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
//  }
//}





// Our custom scheduler, multi-GPU with beacons.
// XXX This algorithm treats memory as a hard requirement, and
// then chooses the GPU with the fewest current warps. It doesn't consider
// thread blocks. It doesn't consider when warp limits might be saturated.
void sched_mgb_basic(void) {  //Alg.3
  int tmp_dev_id;
  int *head_p;
  int *tail_p;
  int *jobs_running_on_gpu;
  int *jobs_waiting_on_gpu;
  int assigned;
  struct timespec ts;
  int boomers_len;
  int i;
  bemps_shm_comm_t *comm;
  int batch_size;

  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;
  jobs_running_on_gpu = &bemps_shm_p->gen->jobs_running_on_gpu;
  jobs_waiting_on_gpu = &bemps_shm_p->gen->jobs_waiting_on_gpu;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
                           &ts);
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);
    bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);

    ALIVE_MSG();

    // First loop: Catch the scheduler's tail back up with the beacon
    // queue's head. If we see a free-beacon, then reclaim that resource.
    //BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
    //BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
    batch_size = 0;
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if (comm->exit_flag) {
        BEMPS_SCHED_LOG("seeing exit flag\n");
        comm->exit_flag = 0;
      } else {
        assert(comm->beacon.mem_B);
        BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
                                                    << "\n");
        if (comm->beacon.mem_B < 0) {
          BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
          stats.num_frees++;
          tmp_dev_id = comm->sched_notif.device_id;
          // Add (don't subtract), because mem_B is negative already
          long tmp_bytes_to_free = comm->beacon.mem_B;
          long tmp_warps_to_free = comm->beacon.warps;
          BEMPS_SCHED_LOG("Freeing " << tmp_bytes_to_free << " bytes "
                          << "from device " << tmp_dev_id << "\n");
          BEMPS_SCHED_LOG("Freeing " << tmp_warps_to_free << " warps "
                          << "from device " << tmp_dev_id << "\n");
          gpus_in_use[tmp_dev_id].mem_B += tmp_bytes_to_free;
          gpus_in_use[tmp_dev_id].warps += tmp_warps_to_free;
          --*jobs_running_on_gpu;
        } else {
          stats.num_beacons++;
          boomers.push_back(comm);
          batch_size++; // batch size doesn't include free() beacons
          ++*jobs_waiting_on_gpu;
        }
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }

    if (batch_size > stats.max_observed_batch_size) {
      stats.max_observed_batch_size = batch_size;
    }

    // Second loop: Walk the boomers. This time handle regular beacons, and
    // attempt to assign them to a device. The boomers are sorted by memory
    // footprint, highest to lowest.
    boomers.sort(mem_footprint_compare);
    boomers_len = boomers.size();
    if (boomers_len > stats.max_len_boomers) {
      stats.max_len_boomers = boomers_len;
    }
    if (boomers_len > 0) {
      BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
    }
    for (i = 0; i < boomers_len; i++) {
      assigned = 0;
      comm = boomers.front();
      boomers.pop_front();

      if (comm->age > stats.max_age) {
        stats.max_age = comm->age;
      }

      // The target device for a process must have memory available for it,
      // and it should be the device with the least warps currently in use.
      long curr_min_warps = LONG_MAX;
      int target_dev_id = 0;
      for (tmp_dev_id = 0; tmp_dev_id < NUM_GPUS; tmp_dev_id++) {
        BEMPS_SCHED_LOG("Checking device " << tmp_dev_id << "\n"
                        << "  Total avail bytes: " << GPUS[tmp_dev_id].mem_B << "\n"
                        << "  In-use bytes: " << gpus_in_use[tmp_dev_id].mem_B << "\n"
                        << "  Trying-to-fit bytes: " << comm->beacon.mem_B << "\n"
                        << "  In-use warps: " << gpus_in_use[tmp_dev_id].warps << "\n"
                        << "  Trying-to-add warps: " << comm->beacon.warps << "\n");
        if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
             GPUS[tmp_dev_id].mem_B)) {
          if (gpus_in_use[tmp_dev_id].warps < curr_min_warps) {
              curr_min_warps = gpus_in_use[tmp_dev_id].warps;
              target_dev_id = tmp_dev_id;
              assigned = 1;
          }
        } else {
            BEMPS_SCHED_LOG("Couldn't fit " << comm->beacon.mem_B << "\n");
        }
      }

      if (!assigned) {
        // FIXME: need to add stats, and possibly a way to reserve a
        // GPU to prevent starving.
        comm->age++;
        boomers.push_back(comm);
        // don't adjust jobs-waiting-on-gpu. it was incremented when job first
        // went into the boomers list
      } else {
        long tmp_bytes_to_add = comm->beacon.mem_B;
        long tmp_warps_to_add = comm->beacon.warps;
        BEMPS_SCHED_LOG("Adding " << tmp_bytes_to_add << " bytes "
                        << "to device " << target_dev_id << "\n");
        BEMPS_SCHED_LOG("Adding " << tmp_warps_to_add << " warps "
                        << "to device " << target_dev_id << "\n");
        gpus_in_use[target_dev_id].mem_B += tmp_bytes_to_add;
        gpus_in_use[target_dev_id].warps += tmp_warps_to_add;
        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
                                            << "on device(" << target_dev_id
                                            << ")\n");
        // FIXME Is this SCHEDULER_READ state helping at all?
        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = target_dev_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
        ++*jobs_running_on_gpu;
        --*jobs_waiting_on_gpu;
      }
    }
    bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
  }
}


void sched_cg(void) {

  int i;
  int rc;
  int device_id;
  int *head_p;
  int *tail_p;
  struct timespec ts;
  bemps_shm_comm_t *comm;
  device_id_count_t *dc;

  for(i = 0; i < NUM_GPUS; i++){
    dc = (device_id_count_t *) malloc(sizeof(device_id_count_t));
    dc->device_id = i;
    dc->count = CG_JOBS_PER_GPU;
    avail_device_id_counts.push_back(dc);
  }

  device_id = 0;
  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    rc = pthread_cond_timedwait(&bemps_shm_p->gen->cond,
                                &bemps_shm_p->gen->lock, &ts);
    //BEMPS_SCHED_LOG("rc from timedwait: " << rc << "\n");
    //BEMPS_SCHED_LOG("strerror of rc: " << strerror(rc) << "\n");
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    //BEMPS_SCHED_LOG("Woke up\n");
    bemps_stopwatch_start(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);

    ALIVE_MSG();

    // catch the scheduler's tail back up with the beacon queue's head
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if (comm->exit_flag) {
        dc = pid_to_device_id_counts[comm->pid];
        BEMPS_SCHED_LOG("pid(" << comm->pid << ") exiting.\n");
        BEMPS_SCHED_LOG("recycling device_id(" << dc->device_id << ").\n");
        dc->count++;
        assert(dc->count <= CG_JOBS_PER_GPU); // error could mean problem with driver
        avail_device_id_counts.sort(avail_devices_compare);
        pid_to_device_id_counts.erase(comm->pid);
        comm->exit_flag = 0;
        comm->pid = 0;
      } else {
        if (pid_to_device_id_counts.find(comm->pid) == pid_to_device_id_counts.end()) {
          // Not found: We're seeing this pid for the first time.
          // This should be a proper beacon (not a free)
          assert(comm->beacon.mem_B > 0);
          dc = avail_device_id_counts.front();
          dc->count--;
          assert(dc->count >= 0); // error could mean a problem with driver
          avail_device_id_counts.sort(avail_devices_compare);
          pid_to_device_id_counts[comm->pid] = dc;
        } else {
          // Found: Do nothing.
          // assert that at least one process (the one that sent this beacon)
          // is assigned to this device (i.e. count should be < max)
          assert(pid_to_device_id_counts[comm->pid]->count < CG_JOBS_PER_GPU);
        }

        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = pid_to_device_id_counts[comm->pid]->device_id;
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
    bemps_stopwatch_end(&sched_stopwatches[SCHED_STOPWATCH_AWAKE]);
  }
}


void sched_single_assignment(void) {
  int i;
  int rc;
  int device_id;
  int *head_p;
  int *tail_p;
  struct timespec ts;
  bemps_shm_comm_t *comm;

  for(i = 0; i < NUM_GPUS; i++){
    avail_device_ids.push_back(i);
  }

  device_id = 0;
  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    rc = pthread_cond_timedwait(&bemps_shm_p->gen->cond,
                                &bemps_shm_p->gen->lock, &ts);
    //BEMPS_SCHED_LOG("rc from timedwait: " << rc << "\n");
    //BEMPS_SCHED_LOG("strerror of rc: " << strerror(rc) << "\n");
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    //BEMPS_SCHED_LOG("Woke up\n");
    ALIVE_MSG();

    // catch the scheduler's tail back up with the beacon queue's head
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      if (comm->exit_flag) {
        BEMPS_SCHED_LOG("pid(" << comm->pid << ") exiting.\n");
        BEMPS_SCHED_LOG("recycling device_id(" << pid_to_device_id[comm->pid]
                        << ").\n");
        avail_device_ids.push_back(pid_to_device_id[comm->pid]);
        pid_to_device_id.erase(comm->pid);
        comm->exit_flag = 0;
        comm->pid = 0;
      } else {
        if (pid_to_device_id.find(comm->pid) == pid_to_device_id.end()) {
          // Not found: We're seeing this pid for the first time.
          // This should be a proper beacon (not a free)
          assert(comm->beacon.mem_B > 0);
          // No avail device could mean there's an issue with the driver
          assert(avail_device_ids.size() > 0);
          pid_to_device_id[comm->pid] = avail_device_ids.back();
          avail_device_ids.pop_back();
        } else {
          // Found: Do nothing.
        }

        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
        comm->sched_notif.device_id = pid_to_device_id[comm->pid];
        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
        sem_post(&comm->sched_notif.sem);
      }

      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
  }
}


//
// Deprecated
//
//void sched_vector(void) {
//  int tmp_dev_id;
//  int *head_p;
//  int *tail_p;
//  int assigned;
//  struct timespec ts;
//  int boomers_len;
//  int i;
//  bemps_shm_comm_t *comm;
//  int batch_size;
//  /*std::priority_queue<bemps_shm_comm_t *,
//                      std::vector<bemps_shm_comm_t *>,
//                      CustomCompare> pq;*/
//  // std::vector<bemps_shm_comm_t *> boomers_sorted;
//
//  head_p = &bemps_shm_p->gen->beacon_q_head;
//  tail_p = &bemps_shm_p->gen->beacon_q_tail;
//
//  while (1) {
//    set_wakeup_time_ns(&ts);
//
//    // wait until we get a signal or time out
//    pthread_mutex_lock(&bemps_shm_p->gen->lock);
//    // TODO spurious wakeups ? shouldn't make a big difference to wake up
//    // randomly from time to time before the batch is ready
//    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
//                           &ts);
//    pthread_mutex_unlock(&bemps_shm_p->gen->lock);
//
//    // First loop: Catch the scheduler's tail back up with the beacon
//    // queue's head. If we see a free-beacon, then reclaim that resource.
//    BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
//    BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
//    batch_size = 0;
//    while (*tail_p != *head_p) {
//      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
//
//      comm = &bemps_shm_p->comm[*tail_p];
//      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
//        // TODO probably want to track a stat for this case
//        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
//                        << "was set. (Not a bug, but unless we're "
//                        << "flooded with beacons, this should be rare."
//                        << "\n");
//        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
//        // FIXME sync shouldn't hurt, but may not help?
//        __sync_synchronize();
//      }
//
//      BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
//                                                  << "\n");
//
//      assert(comm->beacon.mem_B);
//      if (comm->beacon.mem_B < 0) {
//        stats.num_frees++;
//
//        BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
//        tmp_dev_id = comm->sched_notif.device_id;
//        // Add (don't subtract), because mem_B is negative already
//        gpus_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
//        gpus_in_use[tmp_dev_id].warps += comm->beacon.warps;
//      } else {
//        stats.num_beacons++;
//        boomers.push_back(comm);
//        // pq.push(comm);
//        batch_size++;
//      }
//
//      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
//    }
//
//    if (batch_size > stats.max_observed_batch_size) {
//      stats.max_observed_batch_size = batch_size;
//    }
//
//    // Second loop: Walk the boomers. This time handle regular beacons,
//    // and attempt to assign them a device
//    // std::sort(boomers.begin(), boomers.end(), mem_footprint_compare);
//    boomers.sort(mem_footprint_compare);
//    boomers_len = boomers.size();
//    if (boomers_len > stats.max_len_boomers) {
//      stats.max_len_boomers = boomers_len;
//    }
//    BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
//    for (i = 0; i < boomers_len; i++) {
//      assigned = 0;
//      comm = boomers.front();
//      boomers.pop_front();
//
//      if (comm->age > stats.max_age) {
//        stats.max_age = comm->age;
//      }
//
//      for (tmp_dev_id = 0; tmp_dev_id < NUM_GPUS; tmp_dev_id++) {
//        /*if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
//             GPUS[tmp_dev_id].mem_B) &&
//            ((gpus_in_use[tmp_dev_id].warps + comm->beacon.warps) <
//             GPUS[tmp_dev_id].warps)) {*/
//        if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
//             GPUS[tmp_dev_id].mem_B)) {
//          gpus_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
//          gpus_in_use[tmp_dev_id].warps += comm->beacon.warps;
//          assigned = 1;
//          break;
//        }
//      }
//
//      if (!assigned) {
//        // FIXME: need to add stats, and possibly a way to reserve a
//        // GPU to prevent starving.
//        comm->age++;
//        boomers.push_back(comm);
//      } else {
//        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
//                                            << "on device(" << tmp_dev_id
//                                            << ")\n");
//        // FIXME Is this SCHEDULER_READ state helping at all?
//        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
//        comm->sched_notif.device_id = tmp_dev_id;
//        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
//        sem_post(&comm->sched_notif.sem);
//      }
//    }
//  }
//}

//
// Deprecated
//
//void sched_round_robin(void) {
//  int device_id;
//  int tmp_dev_id;
//  int *head_p;
//  int *tail_p;
//  int bcn_idx;
//  int assigned;
//  struct timespec ts;
//  int boomers_len;
//  int i;
//  bemps_shm_comm_t *comm;
//
//  device_id = 0;
//  head_p = &bemps_shm_p->gen->beacon_q_head;
//  tail_p = &bemps_shm_p->gen->beacon_q_tail;
//
//  while (1) {
//    set_wakeup_time_ns(&ts);
//
//    // wait until we get a signal or time out
//    pthread_mutex_lock(&bemps_shm_p->gen->lock);
//    // TODO spurious wakeups ? shouldn't make a big difference to wake up
//    // randomly from time to time before the batch is ready
//    pthread_cond_timedwait(&bemps_shm_p->gen->cond, &bemps_shm_p->gen->lock,
//                           &ts);
//    pthread_mutex_unlock(&bemps_shm_p->gen->lock);
//
//    // Loop zero: Handle old beacons that haven't been scheduled yet.
//    boomers_len = boomers.size();
//    if (boomers_len > stats.max_len_boomers) {
//      stats.max_len_boomers = boomers_len;
//    }
//    BEMPS_SCHED_LOG("boomers_len: " << boomers_len << "\n");
//    for (i = 0; i < boomers_len; i++) {
//      assigned = 0;
//      tmp_dev_id = device_id;
//      comm = boomers.front();
//      boomers.pop_front();
//
//      if (comm->age > stats.max_age) {
//        stats.max_age = comm->age;
//      }
//
//      while (1) {
//        /*if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
//             GPUS[tmp_dev_id].mem_B) &&
//            ((gpus_in_use[tmp_dev_id].warps + comm->beacon.warps) <
//             GPUS[tmp_dev_id].warps)) {*/
//        if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
//             GPUS[tmp_dev_id].mem_B)) {
//          gpus_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
//          gpus_in_use[tmp_dev_id].warps += comm->beacon.warps;
//          assigned = 1;
//          break;
//        }
//
//        tmp_dev_id = (tmp_dev_id + 1) & (NUM_GPUS - 1);
//        if (tmp_dev_id == device_id) {
//          break;
//        }
//      }
//
//      if (!assigned) {
//        // FIXME: need to add stats, and possibly a way to reserve a
//        // GPU to prevent starving.
//        comm->age++;
//        boomers.push_back(comm);
//      } else {
//        // FIXME Is this SCHEDULER_READ state helping at all?
//        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
//        comm->sched_notif.device_id = device_id;
//        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
//        sem_post(&comm->sched_notif.sem);
//
//        device_id = (device_id + 1) & (NUM_GPUS - 1);
//      }
//    }
//
//    // First loop: Catch the scheduler's tail back up with the beacon
//    // queue's head. If we see a free-beacon, then reclaim that resource.
//    bcn_idx = *tail_p;
//    BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");
//    BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
//    while (*tail_p != *head_p) {
//      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
//
//      comm = &bemps_shm_p->comm[*tail_p];
//      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
//        // TODO probably want to track a stat for this case
//        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
//                        << "was set. (Not a bug, but unless we're "
//                        << "flooded with beacons, this should be rare."
//                        << "\n");
//        BEMPS_SCHED_LOG("WARNING: *tail_p: " << (*tail_p) << "\n");
//        // FIXME sync shouldn't hurt, but may not help?
//        __sync_synchronize();
//      }
//
//      BEMPS_SCHED_LOG("First loop seeing mem_B: " << comm->beacon.mem_B
//                                                  << "\n");
//
//      assert(comm->beacon.mem_B);
//      if (comm->beacon.mem_B < 0) {
//        stats.num_frees++;
//
//        BEMPS_SCHED_LOG("Received free-beacon for pid " << comm->pid << "\n");
//        tmp_dev_id = comm->sched_notif.device_id;
//        // Add (don't subtract), because mem_B is negative already
//        gpus_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
//        gpus_in_use[tmp_dev_id].warps += comm->beacon.warps;
//      }
//
//      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
//    }
//
//    // Second loop: Walk the queue again. This time handle regular beacons,
//    // and attempt to assign them a device
//    BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
//    BEMPS_SCHED_LOG("bcn_idx: " << (bcn_idx) << "\n");
//    while (bcn_idx != *tail_p) {
//      BEMPS_SCHED_LOG("Second loop bcn_idx: " << (bcn_idx) << "\n");
//
//      assigned = 0;
//      tmp_dev_id = device_id;
//      comm = &bemps_shm_p->comm[bcn_idx];
//
//      if (comm->beacon.mem_B < 0) {
//        bcn_idx = (bcn_idx + 1) & (BEMPS_BEACON_BUF_SZ - 1);
//        continue;
//      }
//      stats.num_beacons++;
//
//      while (1) {
//        /*if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
//             GPUS[tmp_dev_id].mem_B) &&
//            ((gpus_in_use[tmp_dev_id].warps + comm->beacon.warps) <
//             GPUS[tmp_dev_id].warps)) {*/
//        if (((gpus_in_use[tmp_dev_id].mem_B + comm->beacon.mem_B) <
//             GPUS[tmp_dev_id].mem_B)) {
//          gpus_in_use[tmp_dev_id].mem_B += comm->beacon.mem_B;
//          gpus_in_use[tmp_dev_id].warps += comm->beacon.warps;
//          assigned = 1;
//          BEMPS_SCHED_LOG("  assigned\n");
//          break;
//        }else{
//          BEMPS_SCHED_LOG("  not assigned\n");
//        }
//
//        tmp_dev_id = (tmp_dev_id + 1) & (NUM_GPUS - 1);
//        if (tmp_dev_id == device_id) {
//          break;
//        }
//      }
//
//      if (!assigned) {
//        comm->age++;
//        boomers.push_back(comm);
//      } else {
//        BEMPS_SCHED_LOG("sem_post for pid(" << comm->pid << ") "
//                                            << "on device(" << device_id
//                                            << ")\n");
//        // FIXME Is this SCHEDULER_READ state helping at all?
//        comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
//        comm->sched_notif.device_id = device_id;
//        comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
//        sem_post(&comm->sched_notif.sem);
//
//        device_id = (device_id + 1) & (NUM_GPUS - 1);
//      }
//
//      bcn_idx = (bcn_idx + 1) & (BEMPS_BEACON_BUF_SZ - 1);
//    }
//  }
//}


void sched_no_beacons(int is_round_robin) {
  int rc;
  int device_id;
  int *head_p;
  int *tail_p;
  struct timespec ts;
  bemps_shm_comm_t *comm;

  device_id = 0;
  head_p = &bemps_shm_p->gen->beacon_q_head;
  tail_p = &bemps_shm_p->gen->beacon_q_tail;

  while (1) {
    set_wakeup_time_ns(&ts);

    // wait until we get a signal or time out
    pthread_mutex_lock(&bemps_shm_p->gen->lock);
    // TODO spurious wakeups ? shouldn't make a big difference to wake up
    // randomly from time to time before the batch is ready
    rc = pthread_cond_timedwait(&bemps_shm_p->gen->cond,
                                &bemps_shm_p->gen->lock, &ts);
    BEMPS_SCHED_LOG("rc from timedwait: " << rc << "\n");
    BEMPS_SCHED_LOG("strerror of rc: " << strerror(rc) << "\n");
    pthread_mutex_unlock(&bemps_shm_p->gen->lock);

    BEMPS_SCHED_LOG("Woke up\n");

    // catch the scheduler's tail back up with the beacon queue's head
    while (*tail_p != *head_p) {
      BEMPS_SCHED_LOG("*tail_p: " << (*tail_p) << "\n");
      BEMPS_SCHED_LOG("*head_p: " << (*head_p) << "\n");

      comm = &bemps_shm_p->comm[*tail_p];
      while (comm->state != BEMPS_BEACON_STATE_BEACON_FIRED_E) {
        // TODO probably want to track a stat for this case
        BEMPS_SCHED_LOG("WARNING: Scheduler hit a beacon before FIRED "
                        << "was set. (Not a bug, but unless we're "
                        << "flooded with beacons, this should be rare."
                        << "\n");
        // FIXME sync shouldn't hurt, but may not help?
        __sync_synchronize();
      }

      comm->state = BEMPS_BEACON_STATE_SCHEDULER_READ_E;
      comm->sched_notif.device_id = device_id;
      comm->state = BEMPS_BEACON_STATE_SCHEDULED_E;
      sem_post(&comm->sched_notif.sem);

      //
      // Deprecated
      //
      if (is_round_robin) {
        assert(0);
      }
      //if (is_round_robin) {
      //  device_id = (device_id + 1) & (NUM_GPUS - 1);
      //}
      *tail_p = (*tail_p + 1) & (BEMPS_BEACON_BUF_SZ - 1);
    }
  }
}

void sched(void) {
  if (which_scheduler == SCHED_ALG_ZERO_E) {
    BEMPS_SCHED_LOG("Starting zero scheduler\n");
    sched_no_beacons(0);
  //} else if (which_scheduler == SCHED_ALG_ROUND_ROBIN_E) {
  //  BEMPS_SCHED_LOG("Starting round robin scheduler\n");
  //  sched_no_beacons(1);
  //} else if (which_scheduler == SCHED_ALG_ROUND_ROBIN_BEACONS_E) {
  //  BEMPS_SCHED_LOG("Starting round robin beacons scheduler\n");
  //  sched_round_robin();
  //} else if (which_scheduler == SCHED_ALG_VECTOR_E) {
  //  BEMPS_SCHED_LOG("Starting vector scheduler\n");
  //  sched_vector();
  } else if (which_scheduler == SCHED_ALG_SINGLE_ASSIGNMENT_E) {
    BEMPS_SCHED_LOG("Starting single asssignment scheduler\n");
    sched_single_assignment();
  } else if (which_scheduler == SCHED_ALG_CG_E) {
    BEMPS_SCHED_LOG("Starting C:G scheduler\n");
    BEMPS_SCHED_LOG("  CG_JOBS_PER_GPU: " << CG_JOBS_PER_GPU << "\n");
    sched_cg();
  } else if (which_scheduler == SCHED_ALG_MGB_BASIC_E) {
    BEMPS_SCHED_LOG("Starting mgb basic scheduler\n");
    sched_mgb_basic();
  //} else if (which_scheduler == SCHED_ALG_MGB_SIMPLE_COMPUTE_E) {
  //  BEMPS_SCHED_LOG("Starting mgb simple compute scheduler\n");
  //  sched_mgb_simple_compute();
  } else if (which_scheduler == SCHED_ALG_MGB_E) {
    BEMPS_SCHED_LOG("Starting mgb scheduler\n");
    sched_mgb();
  } else if(which_scheduler == SCHED_ALG_AI_E) {
    BEMPS_SCHED_LOG("Starting ai-heuristic scheduler\n");
    sched_ai_heuristic(RTX_3080Ti_SPECS_AI_FP32_DRAM);
  } else {
    fprintf(stderr, "ERROR: Invalid scheduling algorithm\n");
    exit(2);
  }
  fprintf(stderr, "ERROR: Scheduler loop returned\n");
  exit(3);
}

void parse_args(int argc, char **argv) {
  int i;
  int num_workers;

  max_batch_size = SCHED_DEFAULT_BATCH_SIZE;

  if (argc > 3) {
    usage_and_exit(argv[0]);
  }

  if (argc == 1) {
    usage_and_exit(argv[0]);
  }

  if (strncmp(argv[1], "zero", 5) == 0) {
    which_scheduler = SCHED_ALG_ZERO_E;
  //} else if (strncmp(argv[1], "round-robin", 12) == 0) {
  //  which_scheduler = SCHED_ALG_ROUND_ROBIN_E;
  //} else if (strncmp(argv[1], "round-robin-beacons", 20) == 0) {
  //  which_scheduler = SCHED_ALG_ROUND_ROBIN_BEACONS_E;
  //} else if (strncmp(argv[1], "vector", 7) == 0) {
  //  which_scheduler = SCHED_ALG_VECTOR_E;
  //  max_batch_size = SCHED_VECTOR_BATCH_SIZE;
  } else if (strncmp(argv[1], "single-assignment", 18) == 0) {
    which_scheduler = SCHED_ALG_SINGLE_ASSIGNMENT_E;
  } else if (strncmp(argv[1], "cg", 3) == 0) {
    which_scheduler = SCHED_ALG_CG_E;
    if (argc != 3) {
      usage_and_exit(argv[0]);
    }
    for (i = 0; i < strlen(argv[2]); i++) {
      if (!isdigit(argv[2][i])) {
        usage_and_exit(argv[0]);
      }
    }
    num_workers = atoi(argv[2]);
    CG_JOBS_PER_GPU = num_workers / NUM_GPUS;
    if (num_workers % NUM_GPUS) {
      CG_JOBS_PER_GPU++;
    }
  } else if (strncmp(argv[1], "mgb", 3) == 0) {
    if (strncmp(argv[1], "mgb_basic", 10) == 0) {
      which_scheduler = SCHED_ALG_MGB_BASIC_E;
    //} else if (strncmp(argv[1], "mgb_simple_compute", 19) == 0) {
    //  which_scheduler = SCHED_ALG_MGB_SIMPLE_COMPUTE_E;
    } else if (strncmp(argv[1], "mgb", 4) == 0) {
      which_scheduler = SCHED_ALG_MGB_E;
    } else {
      usage_and_exit(argv[0]);
    }
    max_batch_size = SCHED_MGB_BATCH_SIZE;
    for (i = 0; i < NUM_GPUS; i++) {
      gpus_in_use[i].active_jobs = 0;
      gpus_in_use[i].compute_saturated = 0;
      gpus_in_use[i].mem_B = 0;
      gpus_in_use[i].warps = 0;
      gpus_in_use[i].curr_sm = 0;
      gpus_in_use[i].sms->resize(GPUS[i].num_sms, {
                                   GPUS[i].thread_blocks_per_sm,
                                   GPUS[i].warps_per_sm
                                });
      gpus_by_mem[i].which_gpu = i;
      gpus_by_mem[i].mem_B = &gpus_in_use[i].mem_B;
    }
  } else if (strncmp(argv[1],"ai-heuristic",13)==0) {
      which_scheduler = SCHED_ALG_AI_E;
  } else {
    usage_and_exit(argv[0]);
  }

  if (which_scheduler != SCHED_ALG_CG_E && argc != 2) {
    usage_and_exit(argv[0]);
  }
}

int main(int argc, char **argv) {
  init_gpus();

  BEMPS_SCHED_LOG("BEMPS SCHEDULER\n");
  signal(SIGINT, sigint_handler);

  BEMPS_SCHED_LOG("Parsing args\n");
  parse_args(argc, argv);

  BEMPS_SCHED_LOG("Initializing shared memory.\n");
  bemps_shm_p = bemps_sched_init(max_batch_size);

  p.start_sampling(); // stop is handled in sigint_handler
  sched();

  return 0;
}
