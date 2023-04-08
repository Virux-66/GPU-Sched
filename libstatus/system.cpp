#include <fstream>
#include "system.h"

GPUSystem::GPUSystem(std::string filename): fname(filename), loop(false) {
    NVML_RT_CALL(nvmlInit());

    // Query the num of devices 
    NVML_RT_CALL(nvmlDeviceGetCount(&num_devices));

    // for each device create and GPUDevice object
    for (int i = 0; i < num_devices; i++) {
        devices.push_back(new GPUDevice(i));
    }
}

GPUSystem::~GPUSystem() {
    for (int i = 0; i < num_devices; i++) {
        delete devices[i];
    } 
    NVML_RT_CALL(nvmlShutdown());
}

void GPUSystem::start() {
    loop = true;
    while(loop) {
        std::time_t stamp = std::chrono::high_resolution_clock::now().time_since_epoch( ).count();
        timestamps.push_back(stamp);
        for (int i = 0; i < num_devices; i++) {
            devices[i]->query();
            // print for debug
            nvmlUtilization_t u = devices[i]->get_utilization();
            std::cout << "Device " << i 
                      << ": GPU utilization: " << u.gpu 
                      << ", MEM Utilization: " << u.memory << "\n";
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
}

void GPUSystem::stop() {
    std::this_thread::sleep_for( std::chrono::seconds( 2 ) ); // Retrive a few empty samples
    loop = false;
}

void GPUSystem::print_header(std::ofstream &ofs) {
    ofs << "timestamp";
    for (int i = 0; i < num_devices; i++) ofs << ",device_" << i;
    ofs << "\n";
}

void GPUSystem::dump() {
    std::ofstream core_util_file(fname+"_gpu.csv", std::ios::out);
    std::ofstream mem_util_file(fname+"_mem.csv", std::ios::out);

    print_header(core_util_file);
    print_header(mem_util_file);

    for(int i = 0; i < timestamps.size(); i++) {
        core_util_file << timestamps[i];
        mem_util_file << timestamps[i];
        for (int j = 0; j < num_devices; j++) {
            nvmlUtilization_t u = devices[j]->get_utilization(i);
            core_util_file << "," << u.gpu;
            mem_util_file << "," << u.memory;
        }
        core_util_file << "\n";
        mem_util_file << "\n";
    }
    core_util_file.close();
    mem_util_file.close();
}