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
//
// DIRECTION MAPPING (critical for correctness):
//   dir=0 (horizontal gradient, ~0 deg, edge is vertical)
//         → check left and right neighbours
//   dir=1 (45 deg diagonal, edge runs top-left to bottom-right)
//         → check top-left and bottom-right neighbours (perpendicular to edge)
//   dir=2 (vertical gradient, ~90 deg, edge is horizontal)
//         → check top and bottom neighbours
//   dir=3 (135 deg diagonal, edge runs top-right to bottom-left)
//         → check top-right and bottom-left neighbours (perpendicular to edge)
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
            // Neighbours must be PERPENDICULAR to the edge direction
            switch (d) {
                case 0: // Horizontal gradient (~0 deg), edge is vertical
                        // → check left and right (perpendicular to vertical edge)
                    neighbor1 = mag[y * w + (x - 1)];
                    neighbor2 = mag[y * w + (x + 1)];
                    break;
                case 1: // 45 deg diagonal, edge runs \ (top-left to bottom-right)
                        // → check top-left and bottom-right (perpendicular to edge)
                    neighbor1 = mag[(y - 1) * w + (x - 1)];  // top-left
                    neighbor2 = mag[(y + 1) * w + (x + 1)];  // bottom-right
                    break;
                case 2: // Vertical gradient (~90 deg), edge is horizontal
                        // → check top and bottom (perpendicular to horizontal edge)
                    neighbor1 = mag[(y - 1) * w + x];
                    neighbor2 = mag[(y + 1) * w + x];
                    break;
                case 3: // 135 deg diagonal, edge runs / (top-right to bottom-left)
                        // → check top-right and bottom-left (perpendicular to edge)
                    neighbor1 = mag[(y - 1) * w + (x + 1)];  // top-right
                    neighbor2 = mag[(y + 1) * w + (x - 1)];  // bottom-left
                    break;
            }

            // Keep only if strictly greater than both neighbours
            // (strict > ensures we don't keep flat regions)
            if (center > neighbor1 && center > neighbor2) {
                out[idx] = center;
            } else {
                out[idx] = 0;
            }
        }
    }
}