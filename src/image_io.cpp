#include "image_io.h"
#include <cstdio>
#include <cstdlib>

uint8_t* alloc_image(int w, int h) {
    void* ptr = aligned_alloc(64, (size_t)w * h);
    if (!ptr) {
        fprintf(stderr, "alloc_image: out of memory\n");
        exit(1);
    }
    return (uint8_t*)ptr;
}

uint8_t* load_raw(const char* path, int w, int h) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_raw: cannot open %s\n", path);
        exit(1);
    }
    uint8_t* buf = alloc_image(w, h);
    size_t bytes = fread(buf, 1, (size_t)w * h, f);
    fclose(f);
    if ((int)bytes != w * h) {
        fprintf(stderr, "load_raw: expected %d bytes, got %zu\n", w*h, bytes);
        exit(1);
    }
    return buf;
}

void save_raw(const char* path, const uint8_t* buf, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "save_raw: cannot open %s\n", path);
        exit(1);
    }
    fwrite(buf, 1, (size_t)w * h, f);
    fclose(f);
}