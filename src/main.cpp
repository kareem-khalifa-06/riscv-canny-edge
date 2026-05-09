#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <time.h>

// Returns current time in milliseconds
double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <width> <height> <input.raw> <output_prefix>\n", argv[0]);
        return 1;
    }

    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    const char* in_path = argv[3];
    const char* prefix  = argv[4];

    // --- Load image ---
    uint8_t* src     = load_raw(in_path, w, h);
    uint8_t* blurred = alloc_image(w, h);
    int16_t* Gx      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    int16_t* Gy      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    uint8_t* mag_l1  = alloc_image(w, h);
    uint8_t* mag_l2  = alloc_image(w, h);
    uint8_t* dir     = alloc_image(w, h);

    printf("=== Canny Pipeline Timing ===\n");
    printf("Image: %dx%d\n\n", w, h);

    // --- Stage 1: Gaussian Blur ---
    double t1 = now_ms();
    for (int i = 0; i < 100; i++)
        gaussian_5x5(src, blurred, w, h);
    double t2 = now_ms();
    printf("[Stage 1] Gaussian Blur:    %.3f ms\n", (t2-t1)/100.0);

    // --- Stage 2: Sobel Gradient ---
    double t3 = now_ms();
    for (int i = 0; i < 100; i++)
        sobel(blurred, Gx, Gy, w, h);
    double t4 = now_ms();
    printf("[Stage 2] Sobel Gradient:   %.3f ms\n", (t4-t3)/100.0);

    // --- Stage 3: Magnitude L1 ---
    double t5 = now_ms();
    for (int i = 0; i < 100; i++)
        magnitude_l1(Gx, Gy, mag_l1, w, h);
    double t6 = now_ms();
    printf("[Stage 3] Magnitude L1:     %.3f ms\n", (t6-t5)/100.0);

    // --- Stage 4: Magnitude L2 ---
    double t7 = now_ms();
    for (int i = 0; i < 100; i++)
        magnitude_l2(Gx, Gy, mag_l2, w, h);
    double t8 = now_ms();
    printf("[Stage 4] Magnitude L2:     %.3f ms\n", (t8-t7)/100.0);

    // --- Stage 5: Direction ---
    double t9 = now_ms();
    for (int i = 0; i < 100; i++)
        direction(Gx, Gy, dir, w, h);
    double t10 = now_ms();
    printf("[Stage 5] Direction:        %.3f ms\n", (t10-t9)/100.0);

    // --- Total ---
    double total = (t2-t1 + t4-t3 + t6-t5 + t8-t7 + t10-t9) / 100.0;
    printf("\nTotal pipeline: %.3f ms\n", total);

    // --- Save outputs ---
    char path[256];
    snprintf(path, sizeof(path), "%s_blurred.raw", prefix);
    save_raw(path, blurred, w, h);
    snprintf(path, sizeof(path), "%s_mag_l1.raw", prefix);
    save_raw(path, mag_l1, w, h);
    snprintf(path, sizeof(path), "%s_mag_l2.raw", prefix);
    save_raw(path, mag_l2, w, h);
    snprintf(path, sizeof(path), "%s_dir.raw", prefix);
    save_raw(path, dir, w, h);

    // --- Cleanup ---
    free(src); free(blurred);
    free(Gx);  free(Gy);
    free(mag_l1); free(mag_l2); free(dir);

    printf("Done.\n");
    return 0;
}

