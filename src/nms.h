#pragma once
#include <cstdint>

void non_maximum_suppression(const uint8_t* __restrict__ mag,
                             const uint8_t* __restrict__ dir,
                             uint8_t* __restrict__ out,
                             int w, int h);