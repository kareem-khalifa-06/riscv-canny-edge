#include "gaussian.h"
#include <cstdint>
#include <algorithm>

void gaussian_5x5(const uint8_t* src, uint8_t* dst, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sum = 0;
            for (int ky = 0; ky < 5; ky++) {
                for (int kx = 0; kx < 5; kx++) {
                    int iy = y + ky - 2;
                    int ix = x + kx - 2;
                    // zero-padding: if outside image treat as 0
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                        continue;
                    sum += src[iy * w + ix] * K5[ky][kx];
                }
            }
            dst[y * w + x] = (uint8_t)std::min(sum / 273, 255);
        }
    }
}

