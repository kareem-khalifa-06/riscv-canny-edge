#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <cstdio>
#include <cstdlib>

static uint8_t test_image[48*48];

static void init_test_image() {
    for (int y = 0; y < 48; ++y)
        for (int x = 0; x < 48; ++x)
            test_image[y * 48 + x] = (x >= 24) ? 255 : 0;
}

static void print_row(const uint8_t* buf, int w, int y) {
    for (int x = 20; x < 28; ++x)
        printf("%3d ", buf[y * w + x]);
    printf("\n");
}

int main() {
    const int w = 48, h = 48;
    init_test_image();

    uint8_t* blurred = alloc_image(w, h);
    int16_t* Gx = (int16_t*)aligned_alloc(64, (size_t)w * h * sizeof(int16_t));
    int16_t* Gy = (int16_t*)aligned_alloc(64, (size_t)w * h * sizeof(int16_t));
    uint8_t* mag_l1 = alloc_image(w, h);
    uint8_t* dir = alloc_image(w, h);

    printf("Running pipeline on %dx%d embedded test image...\n", w, h);

    gaussian_5x5(test_image, blurred, w, h);
    sobel(blurred, Gx, Gy, w, h);
    magnitude_l1(Gx, Gy, mag_l1, w, h);
    direction(Gx, Gy, dir, w, h);

    int cy = h / 2;
    printf("\nBlurred center row (cols 20-27): ");
    print_row(blurred, w, cy);
    printf("Magnitude L1 center row:         ");
    print_row(mag_l1, w, cy);
    printf("Direction center row:            ");
    print_row(dir, w, cy);

    uint8_t center_dir = dir[cy * w + (w/2)];
    printf("\nCenter direction: %d (expected 0 for vertical edge)\n", center_dir);

    if (center_dir == 0) {
        printf("TEST PASSED\n");
        return 0;
    } else {
        printf("TEST FAILED\n");
        return 1;
    }
}
