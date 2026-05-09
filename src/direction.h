#pragma once
#include <cstdint>

// Output: 0=0°, 1=45°, 2=90°, 3=135°
void direction(const int16_t* Gx, const int16_t* Gy, uint8_t* dir, int w, int h);

