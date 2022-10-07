
# How to use the code
1. create an Profiler object with test run id, .e.g, Profiler p("test");
2. at the begining of the job, instrument the code with p.start_sampling()
3. at the end of the job, instrument the code with p.end_sampling()

two csv files, named test_gpu.csv and test_mem.csv will be generated, and will be used to plot utilization figures. 

An example can be found in example.cpp

# Example

Building and running:

    $ nvcc -o example example.cpp device.cpp system.cpp profiler.cpp -lnvidia-ml -lcublas  -lcurand
    $ ./example

Alternatively:

    $ make
    $ run.sh
