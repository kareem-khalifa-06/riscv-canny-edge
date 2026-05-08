#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <width> <height> <input.raw> <output_prefix>\n", argv[0]);
        fprintf(stderr, "Example: %s 512 512 lena.raw out\n", argv[0]);
        return 1;
    }

    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    const char* in_path = argv[3];
    const char* prefix = argv[4];

    if (w <= 0 || h <= 0) {
        fprintf(stderr, "Error: width and height must be positive\n");
        return 1;
    }

    uint8_t* src = load_raw(in_path, w, h);
    uint8_t* blurred = alloc_image(w, h);
    int16_t* Gx = (int16_t*)aligned_alloc(64, (size_t)w * h * sizeof(int16_t));
    int16_t* Gy = (int16_t*)aligned_alloc(64, (size_t)w * h * sizeof(int16_t));
    uint8_t* mag_l1 = alloc_image(w, h);
    uint8_t* mag_l2 = alloc_image(w, h);
    uint8_t* dir = alloc_image(w, h);

    if (!Gx || !Gy) {
        fprintf(stderr, "Error: failed to allocate gradient buffers\n");
        return 1;
    }

    printf("Running pipeline on %dx%d image...\n", w, h);

    printf("  Gaussian blur...\n");
    gaussian_5x5(src, blurred, w, h);

    printf("  Sobel gradients...\n");
    sobel(blurred, Gx, Gy, w, h);

    printf("  Magnitude L1...\n");
    magnitude_l1(Gx, Gy, mag_l1, w, h);

    printf("  Magnitude L2...\n");
    magnitude_l2(Gx, Gy, mag_l2, w, h);

    printf("  Direction...\n");
    direction(Gx, Gy, dir, w, h);

    char path[256];

    snprintf(path, sizeof(path), "%s_blurred.raw", prefix);
    save_raw(path, blurred, w, h);
    printf("  Saved: %s\n", path);

    snprintf(path, sizeof(path), "%s_mag_l1.raw", prefix);
    save_raw(path, mag_l1, w, h);
    printf("  Saved: %s\n", path);

    snprintf(path, sizeof(path), "%s_mag_l2.raw", prefix);
    save_raw(path, mag_l2, w, h);
    printf("  Saved: %s\n", path);

    snprintf(path, sizeof(path), "%s_dir.raw", prefix);
    save_raw(path, dir, w, h);
    printf("  Saved: %s\n", path);

    free(src);
    free(blurred);
    free(Gx);
    free(Gy);
    free(mag_l1);
    free(mag_l2);
    free(dir);

    printf("Done.\n");
    return 0;
}
