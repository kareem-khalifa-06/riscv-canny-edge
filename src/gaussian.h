#pragma once
#include <cstdint>
#include <algorithm>

// The 5x5 Gaussian kernel coefficients (sum = 273)
static const int16_t K5[5][5] = {
    { 2,  4,  5,  4,  2},
    { 4,  9, 12,  9,  4},
    { 5, 12, 15, 12,  5},
    { 4,  9, 12,  9,  4},
    { 2,  4,  5,  4,  2}
};

// Generic 5x5 Gaussian blur
// PixelT  = type of input/output pixels (e.g. uint8_t)
// AccumT  = type of accumulator (e.g. int32_t) — must be large enough to avoid overflow
// KernT   = type of kernel coefficients (e.g. int16_t)
// Boundary handling: zero-padding (out-of-bounds pixels treated as 0)
template<typename PixelT, typename AccumT, typename KernT>
void gaussian_5x5(const PixelT* src, PixelT* dst, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            AccumT sum = 0;
            for (int ky = 0; ky < 5; ky++) {
                for (int kx = 0; kx < 5; kx++) {
                    int iy = y + ky - 2;
                    int ix = x + kx - 2;
                    // zero-padding: if outside image treat as 0
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                        continue;
                    sum += (AccumT)src[iy * w + ix] * (AccumT)K5[ky][kx];
                }
            }
            // divide by 273, clamp to [0, 255]
            dst[y * w + x] = (PixelT)std::min(
                (AccumT)(sum / 273),
                (AccumT)255
            );
        }
    }
}

// Convenience wrapper using standard types for grayscale images
// This is what most code should call
inline void gaussian_5x5(const uint8_t* src, uint8_t* dst, int w, int h) {
    gaussian_5x5<uint8_t, int32_t, int16_t>(src, dst, w, h);
}

