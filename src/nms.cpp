#include "nms.h"
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// non_maximum_suppression
//
// For each interior pixel, compare its magnitude against its two neighbours
// along the gradient direction.  If it is not the maximum of the three,
// suppress it (set to 0).  Otherwise keep the original magnitude.
//
// Boundary pixels (1-pixel border) are always suppressed because one or
// both neighbours would be outside the image.
// ─────────────────────────────────────────────────────────────────────────────
void non_maximum_suppression(const uint8_t* __restrict__ mag,
                             const uint8_t* __restrict__ dir,
                             uint8_t* __restrict__ out,
                             int w, int h)
{
    // Zero the entire output first (handles the 1-pixel border)
    memset(out, 0, (size_t)w * h);

    // Only process interior pixels where both neighbours in any direction exist
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const int idx = y * w + x;
            const uint8_t center = mag[idx];
            const uint8_t d = dir[idx];

            uint8_t neighbor1 = 0;
            uint8_t neighbor2 = 0;

            // Select neighbours based on gradient direction
            switch (d) {
                case 0: // Horizontal gradient → check left and right
                    neighbor1 = mag[y * w + (x - 1)];
                    neighbor2 = mag[y * w + (x + 1)];
                    break;
                case 1: // 45-degree diagonal → check top-right and bottom-left
                    neighbor1 = mag[(y - 1) * w + (x + 1)];
                    neighbor2 = mag[(y + 1) * w + (x - 1)];
                    break;
                case 2: // Vertical gradient → check top and bottom
                    neighbor1 = mag[(y - 1) * w + x];
                    neighbor2 = mag[(y + 1) * w + x];
                    break;
                case 3: // 135-degree diagonal → check top-left and bottom-right
                    neighbor1 = mag[(y - 1) * w + (x - 1)];
                    neighbor2 = mag[(y + 1) * w + (x + 1)];
                    break;
            }

            // Keep only if strictly greater than both neighbours
            // (strict >  ensures we don't keep flat regions)
            if (center > neighbor1 && center > neighbor2) {
                out[idx] = center;
            } else {
                out[idx] = 0;
            }
        }
    }
}
