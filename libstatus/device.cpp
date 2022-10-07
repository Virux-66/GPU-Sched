#include "device.h"

void GPUDevice::query() {
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(device, &util);
    utilizations.push_back(util);
}

nvmlUtilization_t GPUDevice::get_utilization(int i) {
    nvmlUtilization_t util;
    if (i > utilizations.size()) {
        util.gpu = 0;
        util.memory = 0;
    } else { 
        util = utilizations[i];
    }
    return util;
}

nvmlUtilization_t GPUDevice::get_utilization() {
    nvmlUtilization_t util;
    if (0 == utilizations.size()) {
        util.gpu = 0;
        util.memory = 0;
    } else { 
        util = utilizations.back();
    }
    return util;
}