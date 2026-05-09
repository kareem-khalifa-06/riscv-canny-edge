#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// Generate a completely black image
void gen_black(uint8_t* buf, int w, int h) {
    memset(buf, 0, w * h);
}

// Generate a completely white image
void gen_white(uint8_t* buf, int w, int h) {
    memset(buf, 255, w * h);
}

// Generate uniform grey image (all pixels = value)
void gen_uniform(uint8_t* buf, int w, int h, uint8_t value) {
    memset(buf, value, w * h);
}

// Generate vertical edge: left half black, right half white
// Used to test Sobel X (should give large Gx, small Gy)
void gen_vertical_edge(uint8_t* buf, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            buf[y * w + x] = (x < w / 2) ? 0 : 255;
}

// Generate horizontal edge: top half black, bottom half white
// Used to test Sobel Y (should give large Gy, small Gx)
void gen_horizontal_edge(uint8_t* buf, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            buf[y * w + x] = (y < h / 2) ? 0 : 255;
}

// Generate diagonal edge: above diagonal = black, below = white
void gen_diagonal_edge(uint8_t* buf, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            buf[y * w + x] = (x > y) ? 255 : 0;
}

// Generate white rectangle on black background
void gen_rectangle(uint8_t* buf, int w, int h) {
    memset(buf, 0, w * h);
    int rx = w / 4, ry = h / 4;
    int rw = w / 2, rh = h / 2;
    for (int y = ry; y < ry + rh; y++)
        for (int x = rx; x < rx + rw; x++)
            buf[y * w + x] = 255;
}

// Generate single bright pixel in the middle (impulse)
// Used to test Gaussian spread
void gen_impulse(uint8_t* buf, int w, int h) {
    memset(buf, 0, w * h);
    buf[(h / 2) * w + (w / 2)] = 255;
}

void save_raw(const char* path, uint8_t* buf, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("ERROR: cannot save %s\n", path); return; }
    fwrite(buf, 1, w * h, f);
    fclose(f);
    printf("Saved: %s\n", path);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <width> <height>\n", argv[0]);
        printf("Generates test images in current directory\n");
        return 1;
    }

    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    uint8_t* buf = (uint8_t*)malloc(w * h);

    gen_black(buf, w, h);
    save_raw("test_black.raw", buf, w, h);

    gen_white(buf, w, h);
    save_raw("test_white.raw", buf, w, h);

    gen_uniform(buf, w, h, 128);
    save_raw("test_uniform128.raw", buf, w, h);

    gen_vertical_edge(buf, w, h);
    save_raw("test_vertical_edge.raw", buf, w, h);

    gen_horizontal_edge(buf, w, h);
    save_raw("test_horizontal_edge.raw", buf, w, h);

    gen_diagonal_edge(buf, w, h);
    save_raw("test_diagonal_edge.raw", buf, w, h);

    gen_rectangle(buf, w, h);
    save_raw("test_rectangle.raw", buf, w, h);

    gen_impulse(buf, w, h);
    save_raw("test_impulse.raw", buf, w, h);

    free(buf);
    printf("Done! Generated 8 test images at %dx%d\n", w, h);
    return 0;
}

