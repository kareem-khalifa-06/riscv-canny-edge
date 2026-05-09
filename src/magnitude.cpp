#include "magnitude.h"
#include <cmath>
#include <cstdlib>

void magnitude_l1(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h) {
    const int n = w * h;
    int32_t max_val = 0;

    // Pass 1: compute raw L1 magnitudes and find max
    for (int i = 0; i < n; ++i) {
        int32_t m = std::abs(Gx[i]) + std::abs(Gy[i]);
        if (m > max_val) max_val = m;
    }
    if (max_val == 0) max_val = 1;

    // Pass 2: normalize to [0, 255]
    for (int i = 0; i < n; ++i) {
        int32_t m = std::abs(Gx[i]) + std::abs(Gy[i]);
        mag[i] = static_cast<uint8_t>((m * 255) / max_val);
    }
}

void magnitude_l2(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h) {
    const int n = w * h;
    int32_t max_val = 0;

    // Pass 1: compute raw L2 magnitudes and find max
    for (int i = 0; i < n; ++i) {
        int32_t gx = Gx[i];
        int32_t gy = Gy[i];
        int32_t m = static_cast<int32_t>(std::sqrt(gx * gx + gy * gy));
        if (m > max_val) max_val = m;
    }
    if (max_val == 0) max_val = 1;

    // Pass 2: normalize to [0, 255]
    for (int i = 0; i < n; ++i) {
        int32_t gx = Gx[i];
        int32_t gy = Gy[i];
        int32_t m = static_cast<int32_t>(std::sqrt(gx * gx + gy * gy));
        mag[i] = static_cast<uint8_t>((m * 255) / max_val);
    }
}

