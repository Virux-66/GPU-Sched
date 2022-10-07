#include "profiler.h"

std::thread Profiler::start() {
    std::thread start_thread(&GPUSystem::start, &target);
    return start_thread;
}

void Profiler::start_sampling() {
    profiling_thread = start();
}

void Profiler::stop_sampling() {
    std::thread kill_thread(&GPUSystem::stop, &target);
    profiling_thread.join();
    kill_thread.join();
    target.dump();
}
