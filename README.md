
GPU-Sched is a collection of libraries and a scheduler. The libraries include
LLVM passes, as well as runtime libraries for programs built with these passes.
The scheduler is a single binary that includes a few prototypes for interacting
with applications bound for the GPU, and which could benefit from sharing.

Jiabin Zheng update:
We implement **AIGPU**, an arithmetic-intensity-guided single-GPU scheduling model,
based on this framework. In short,  **AIGPU** evaluates the degree of overlapped 
execution of multiple kernels to be scheduled. Then, it selects a concurrent 
kernel set whose aggregate arithmetic intensity approaches the x-coordinate of 
the ridge point of the GPU. Such a concurrent kernel set is able to utilize GPU's
compute and bandwidth resources more efficiently so as to improve the overall throughput.


# Requirements
* cmake (tested with 3.25.0)
* llvm (tested with 9.0.0)
* cuda (tested with 11.7)
* OR-Tools (tested with 9.4) (https://github.com/google/or-tools) (Updated)
* cs-roofline-toolkit (https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/roofline-hackathon-2020) (Update)
* libstatus (https://github.com/portersrc/libstatus)

Jiabin Zheng update:
Note: The original system for development was based off Ubuntu 18.04.
Our experiment was tested on Ubuntu 20.04 LTS. In addition, you should
follow up the instructions in (https://github.com/google/or-tools/blob/stable/cmake/README.md)
to build OR-Tools as an standalone. We use cs-roofline-toolkit to measure
the RTX 3080Ti's Roofline model in which we can find the x-coordinate
of the ridge piont. 

# Building
    $ git clone https://github.com/Virux-66/GPU-Sched.git(Updated)
    $ cd GPU-sched
    $ mkdir build
    $ cd build
    $ cmake ../src
    $ cmake --build . 
Note: Ensure libstatus.so is on your LD\_LIBRARY\_PATH.

# Compiling the benchmarks

 Before compiling the benchmarks, you should replace $BEMPS_BUILD_DIR in Benchmarks_kernel(block)-level/common/make.config and the framework build directory. Then you should enter GPU-Sched/Benchmarks_kernel-level and GPU-Sched/Benchmarks_block-level then run:
    $ make


# Compiling applications with GPU-Sched's passes

One Rodinia example for compiling with the GPU-Sched toolchain is backprop:

    Benchmarks/rodinia_cuda_3.1/cuda/backprop/Makefile
    Benchmarks/rodinia_cuda_3.1/common/make.config

For the darknet example, refer to 

    Benchmarks/darknet/Makefile

In brief, you'll need to build with the libWrapperPass or libGPUBeaconPass
(either with opt or clang), e.g.

    opt -load libWrapperPass.so -WP <foo.bc >foo_mod.bc

or:

    clang -Xclang -load -Xclang libGPUBeaconPass.so foo.ii -c

And you'll need to link with the lazy runtime library and bemps:

    clang -llazy -lbemps


# Running the scheduler

    $ ./bemps_sched -h

    Usage:
        ./bemps_sched <which_scheduler> [jobs_per_gpu] <num_of_gpu>

        which_scheduler is one of:
          zero, single-assignment, cg, mgb_basic, mgb, ai-heuristic, ai-mgb_basic

        jobs_per_gpu is required and only valid for cg; it is an int that
        specifies the maximum number of jobs that can be run a GPU

        num_of_gpu is one of:
            uni-gpu, multi-gpu


# Example

Terminal 1:

    $ cd example
    $ ../build/runtime/sched/bemps_sched zero

Terminal 2:

    $ cd example
    $ LD_LIBRARY_PATH=../build/runtime/bemps bash workload.sh
