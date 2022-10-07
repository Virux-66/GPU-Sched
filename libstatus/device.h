#ifndef _DEVICE_H_
#define _DEVICE_H_

#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>
#include <nvml.h>

#include "common.h"

int constexpr device_name_length = 64;
class GPUDevice {
private:
    nvmlDevice_t device; // device hanlder
    char name[device_name_length]; // device name
    std::vector<nvmlUtilization_t> utilizations; // contains samplings of utilizations (Cores and Memory)
public:
    GPUDevice(int device_id) {
        NVML_RT_CALL(nvmlDeviceGetHandleByIndex_v2(device_id, &device));
        NVML_RT_CALL(nvmlDeviceGetName(device, name, device_name_length));
        std::cout << "Device " << device_id << "(" << device << ") : " << name << "\n";
    }

    void query();
    nvmlUtilization_t get_utilization(int i); // get both gpu and memory utilization
    nvmlUtilization_t get_utilization(); // get both gpu and memory utilization

};

#endif // _DEVICE_H_
