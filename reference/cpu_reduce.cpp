#include "reference/cpu_reduce.h"

float CpuReduceSumFloatOrder(const std::vector<float>& input) {
    float sum = 0.0f;
    for (float v : input) {
        sum += v;
    }
    return sum;
}

double CpuReduceSumDouble(const std::vector<float>& input) {
    double sum = 0.0;
    for (float v : input) {
        sum += static_cast<double>(v);
    }
    return sum;
}
