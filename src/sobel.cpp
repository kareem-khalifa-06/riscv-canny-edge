#include "sobel.h"

void sobel(const uint8_t* src, int16_t* Gx, int16_t* Gy, int w, int h) {
    // Sobel X kernel: detects vertical edges (horizontal gradient)
    const int kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // Sobel Y kernel: detects horizontal edges (vertical gradient)
    const int ky[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int sum_x = 0;
            int sum_y = 0;

            for (int ky_idx = -1; ky_idx <= 1; ++ky_idx) {
                for (int kx_idx = -1; kx_idx <= 1; ++kx_idx) {
                    int py = y + ky_idx;
                    int px = x + kx_idx;

                    // Zero-padding: skip out-of-bounds pixels (treat as 0)
                    if (py >= 0 && py < h && px >= 0 && px < w) {
                        uint8_t pixel = src[py * w + px];
                        sum_x += pixel * kx[ky_idx + 1][kx_idx + 1];
                        sum_y += pixel * ky[ky_idx + 1][kx_idx + 1];
                    }
                }
            }

            Gx[y * w + x] = static_cast<int16_t>(sum_x);
            Gy[y * w + x] = static_cast<int16_t>(sum_y);
        }
    }
}

