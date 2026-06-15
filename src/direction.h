#pragma once
#include <cstdint>

// Output: 0=0deg, 1=45deg, 2=90deg, 3=135deg
void direction(const int16_t* Gx, const int16_t* Gy, uint8_t* dir, int w, int h);

// RVV vectorised direction (only available when compiling with -march=rv64gcv)
#ifdef __riscv_v
void direction_rvv(const int16_t* __restrict__ Gx,
                   const int16_t* __restrict__ Gy,
                   uint8_t*  __restrict__ dir,
                   int w, int h);
#endif
