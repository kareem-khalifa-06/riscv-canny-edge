#pragma once
#include <cstdint>

// Computes Sobel gradients Gx and Gy from src
// Gx detects vertical edges, Gy detects horizontal edges
// Output: two separate int16_t arrays (SoA layout)
void sobel(const uint8_t* src, int16_t* Gx, int16_t* Gy, int w, int h);

