#pragma once
#include <cstdint>

// The 5x5 Gaussian kernel coefficients (sum = 273)
static const int16_t K5[5][5] = {
    { 2,  4,  5,  4,  2},
    { 4,  9, 12,  9,  4},
    { 5, 12, 15, 12,  5},
    { 4,  9, 12,  9,  4},
    { 2,  4,  5,  4,  2}
};

// Applies 5x5 Gaussian blur to src, writes result to dst
// Boundary handling: zero-padding
void gaussian_5x5(const uint8_t* src, uint8_t* dst, int w, int h);

