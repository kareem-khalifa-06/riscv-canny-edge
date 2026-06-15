#pragma once
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Double Thresholding with Hysteresis Edge Tracing
//
// Classifies each pixel into one of three categories:
//   STRONG_EDGE (255) : mag >= high_thresh  → always kept
//   WEAK_EDGE   (128) : low_thresh <= mag < high_thresh  → kept only if
//                       connected to a strong edge (8-connected neighbourhood)
//   SUPPRESSED  (0)   : mag < low_thresh  → always discarded
//
// The hysteresis pass walks weak pixels and promotes any that are
// 8-connected to at least one strong pixel.  This prevents broken edges
// while suppressing noise.
//
// Typical threshold ratio: low = 0.05 * max_mag, high = 0.15 * max_mag
// (or user-supplied absolute values).
// ─────────────────────────────────────────────────────────────────────────────

// Output codes
static constexpr uint8_t EDGE_STRONG  = 255;
static constexpr uint8_t EDGE_WEAK    = 128;
static constexpr uint8_t EDGE_NONE    = 0;

// ── Double thresholding ─────────────────────────────────────────────────────
//   Classifies pixels as strong, weak, or suppressed based on magnitude.
//   No neighbourhood analysis yet — that happens in hysteresis().
// ─────────────────────────────────────────────────────────────────────────────
void double_threshold(const uint8_t* __restrict__ mag,
                      uint8_t* __restrict__ out,
                      int w, int h,
                      uint8_t low_thresh,
                      uint8_t high_thresh);

// ── Hysteresis edge tracing ─────────────────────────────────────────────────
//   Scans the image and promotes weak pixels (EDGE_WEAK) to strong
//   (EDGE_STRONG) if they are 8-connected to at least one strong pixel.
//   Repeats until no more promotions occur (connected-component propagation).
//
//   After this pass, remaining weak pixels are demoted to EDGE_NONE.
// ─────────────────────────────────────────────────────────────────────────────
void hysteresis(uint8_t* __restrict__ img, int w, int h);

// ── Convenience: threshold + hysteresis in one call ─────────────────────────
inline void threshold_and_hysteresis(const uint8_t* __restrict__ mag,
                                      uint8_t* __restrict__ out,
                                      int w, int h,
                                      uint8_t low_thresh,
                                      uint8_t high_thresh)
{
    double_threshold(mag, out, w, h, low_thresh, high_thresh);
    hysteresis(out, w, h);
}
