#ifndef _PROFILER_H_
#define _PROFILER_H_

#include <thread>
#include "system.h"

class Profiler {
private:
    std::string fname;
    std::thread profiling_thread;
    GPUSystem target;

    std::thread start();

public:
    Profiler(char *log_file_name): fname(log_file_name), target(fname) {}
    void start_sampling(); 
    void stop_sampling(); 

};

#endif // _PROFILER_H_