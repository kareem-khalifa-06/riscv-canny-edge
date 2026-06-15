#include "threshold.h"
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// double_threshold
//
// First pass of Canny edge classification:
//   mag >= high_thresh  → EDGE_STRONG (255)
//   low_thresh <= mag < high_thresh → EDGE_WEAK (128)
//   mag < low_thresh    → EDGE_NONE (0)
//
// The thresholds are typically derived from the maximum magnitude:
//   high = (uint8_t)(max_mag * 0.15)
//   low  = (uint8_t)(max_mag * 0.05)
// but can also be set manually for reproducible tests.
// ─────────────────────────────────────────────────────────────────────────────
void double_threshold(const uint8_t* __restrict__ mag,
                      uint8_t* __restrict__ out,
                      int w, int h,
                      uint8_t low_thresh,
                      uint8_t high_thresh)
{
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        const uint8_t m = mag[i];
        if (m >= high_thresh) {
            out[i] = EDGE_STRONG;
        } else if (m >= low_thresh) {
            out[i] = EDGE_WEAK;
        } else {
            out[i] = EDGE_NONE;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// hysteresis
//
// Second pass: trace edges from strong pixels through weak neighbours.
//
// Algorithm:
//   1. For every weak pixel, check its 8-connected neighbours.
//   2. If any neighbour is STRONG, promote the weak pixel to STRONG.
//   3. Repeat step 1-2 until a full pass produces no new promotions.
//      (This propagates the strong label along edge chains.)
//   4. Any remaining weak pixels are demoted to NONE — they are isolated
//      noise, not part of a real edge.
//
// This is a flood-fill / connected-component propagation.  We use a
// repeated-scan approach rather than a queue because:
//   - The image is small (typical embedded use: <= 1 MP)
//   - Repeated scans are cache-friendly and branch-predictor friendly
//   - Worst-case iterations = longest edge chain in pixels, which is
//     bounded by image diagonal (~1500 for 1080p)
// ─────────────────────────────────────────────────────────────────────────────
void hysteresis(uint8_t* __restrict__ img, int w, int h)
{
    bool changed = true;
    int iterations = 0;

    // Phase 1: propagate strong connectivity through weak pixels
    while (changed) {
        changed = false;
        ++iterations;

        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const int idx = y * w + x;

                // Only weak pixels can be promoted
                if (img[idx] != EDGE_WEAK)
                    continue;

                // Check 8-connected neighbours for any strong pixel
                bool connected_to_strong = false;
                for (int dy = -1; dy <= 1 && !connected_to_strong; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0)
                            continue;
                        const int ny = y + dy;
                        const int nx = x + dx;
                        if (img[ny * w + nx] == EDGE_STRONG) {
                            connected_to_strong = true;
                            break;
                        }
                    }
                }

                if (connected_to_strong) {
                    img[idx] = EDGE_STRONG;
                    changed = true;
                }
            }
        }
    }

    // Phase 2: demote all remaining weak pixels to NONE
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        if (img[i] == EDGE_WEAK) {
            img[i] = EDGE_NONE;
        }
    }
}
