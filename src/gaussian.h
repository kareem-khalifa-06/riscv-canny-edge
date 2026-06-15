#pragma once
#include <cstdint>
#include <algorithm>
#include <limits>

static const int16_t K5[5][5] = {
    { 1,  4,  7,  4,  1},
    { 4, 16, 26, 16,  4},
    { 7, 26, 41, 26,  7},
    { 4, 16, 26, 16,  4},
    { 1,  4,  7,  4,  1}
};

template<typename PixelT, typename AccumT, typename KernT>
void gaussian_5x5(const PixelT* src, PixelT* dst, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            AccumT sum = 0;
            for (int ky = 0; ky < 5; ky++) {
                for (int kx = 0; kx < 5; kx++) {
                    int iy = y + ky - 2;
                    int ix = x + kx - 2;
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                        continue;
                    sum += (AccumT)src[iy * w + ix] * (AccumT)K5[ky][kx];
                }
            }
            AccumT result = (sum + (AccumT)136) / (AccumT)273;
            AccumT pix_max = (AccumT)std::numeric_limits<PixelT>::max();
            if (result > pix_max) result = pix_max;
            if (result < (AccumT)0) result = (AccumT)0;
            dst[y * w + x] = (PixelT)result;
        }
    }
}

inline void gaussian_5x5(const uint8_t* src, uint8_t* dst, int w, int h) {
    gaussian_5x5<uint8_t, int32_t, int16_t>(src, dst, w, h);
}

#ifdef __riscv_v
extern "C" void gaussian_5x5_rvv(const uint8_t* src, uint8_t* dst, int w, int h);
#endif