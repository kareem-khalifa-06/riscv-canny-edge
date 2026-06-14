#pragma once
#include <cstdint>

void magnitude_l1(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h);
void magnitude_l2(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h);
void magnitude_l1_rvv(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h);