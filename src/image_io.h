#pragma once
#include <cstdint>
#include <cstdlib>

// Allocate a 64-byte aligned image buffer (required for RVV loads)
uint8_t* alloc_image(int w, int h);

// Load a raw grayscale image (exactly w*h bytes, no header)
uint8_t* load_raw(const char* path, int w, int h);

// Save a raw grayscale image to disk
void save_raw(const char* path, const uint8_t* buf, int w, int h);