#include "direction.h"
#include <cstdlib>

void direction(const int16_t* Gx, const int16_t* Gy, uint8_t* dir, int w, int h) {
    const int n = w * h;

    for (int i = 0; i < n; ++i) {
        int32_t gx = Gx[i];
        int32_t gy = Gy[i];
        int32_t ax = std::abs(gx);
        int32_t ay = std::abs(gy);

        uint8_t d;
        // tan(22.5°) ≈ 2/5, tan(67.5°) ≈ 12/5
        if (ay * 5 < ax * 2) {
            d = 0;                      // ~0°, edge is vertical
        } else if (ay * 5 > ax * 12) {
            d = 2;                      // ~90°, edge is horizontal
        } else {
            // Diagonal
            if ((gx >= 0 && gy >= 0) || (gx < 0 && gy < 0))
                d = 1;                  // 45°
            else
                d = 3;                  // 135°
        }
        dir[i] = d;
    }
}

