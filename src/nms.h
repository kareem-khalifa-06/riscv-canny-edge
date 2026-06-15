#pragma once
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Non-Maximum Suppression (NMS)
//
// Thins edges by keeping only pixels that are local maxima along the
// gradient direction.  For each pixel, inspect the two neighbours in the
// direction of the gradient (opposite directions).  If the current pixel
// is not strictly greater than both neighbours, suppress it to 0.
//
// Direction encoding (from direction.cpp):
//   0 = ~0 deg   (gradient is horizontal) → check left/right neighbours
//   1 = ~45 deg  → check diagonal (top-right / bottom-left)
//   2 = ~90 deg  (gradient is vertical)   → check top/bottom neighbours
//   3 = ~135 deg → check diagonal (top-left / bottom-right)
//
// Boundary pixels (where a neighbour would fall outside the image) are
// always suppressed to 0 — this matches OpenCV behaviour.
// ─────────────────────────────────────────────────────────────────────────────
void non_maximum_suppression(const uint8_t* __restrict__ mag,
                             const uint8_t* __restrict__ dir,
                             uint8_t* __restrict__ out,
                             int w, int h);
