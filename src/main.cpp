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

// RVV function declarations (defined in rvv/)
#ifdef __riscv_v
void gaussian_5x5_rvv(const uint8_t* src, uint8_t* dst, int w, int h);
void sobel_rvv(const uint8_t* src, int16_t* Gx, int16_t* Gy, int w, int h);
void magnitude_l1_rvv(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h);
#endif

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

    uint8_t* src     = load_raw(in_path, w, h);
    uint8_t* blurred = alloc_image(w, h);
    int16_t* Gx      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    int16_t* Gy      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    uint8_t* mag_l1  = alloc_image(w, h);
    uint8_t* mag_l2  = alloc_image(w, h);
    uint8_t* dir     = alloc_image(w, h);

    printf("=== Canny Pipeline Timing ===\n");
    printf("Image: %dx%d\n\n", w, h);

    // ── Scalar stages ────────────────────────────────────────────────────────

    double t1 = now_ms();
    for (int i = 0; i < 100; i++)
        gaussian_5x5(src, blurred, w, h);
    double t2 = now_ms();
    printf("[Scalar] Gaussian Blur:    %.3f ms\n", (t2-t1)/100.0);

    double t3 = now_ms();
    for (int i = 0; i < 100; i++)
        sobel(blurred, Gx, Gy, w, h);
    double t4 = now_ms();
    printf("[Scalar] Sobel Gradient:   %.3f ms\n", (t4-t3)/100.0);

    double t5 = now_ms();
    for (int i = 0; i < 100; i++)
        magnitude_l1(Gx, Gy, mag_l1, w, h);
    double t6 = now_ms();
    printf("[Scalar] Magnitude L1:     %.3f ms\n", (t6-t5)/100.0);

    double t7 = now_ms();
    for (int i = 0; i < 100; i++)
        magnitude_l2(Gx, Gy, mag_l2, w, h);
    double t8 = now_ms();
    printf("[Scalar] Magnitude L2:     %.3f ms\n", (t8-t7)/100.0);

    double t9 = now_ms();
    for (int i = 0; i < 100; i++)
        direction(Gx, Gy, dir, w, h);
    double t10 = now_ms();
    printf("[Scalar] Direction:        %.3f ms\n", (t10-t9)/100.0);

    double scalar_total = (t2-t1 + t4-t3 + t6-t5 + t8-t7 + t10-t9) / 100.0;
    printf("\n[Scalar] Total pipeline:   %.3f ms\n", scalar_total);

    // ── RVV stages ───────────────────────────────────────────────────────────

#ifdef __riscv_v
    uint8_t* blurred_rvv = alloc_image(w, h);
    int16_t* Gx_rvv      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    int16_t* Gy_rvv      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    uint8_t* mag_rvv     = alloc_image(w, h);

    printf("\n");

    double r1 = now_ms();
    for (int i = 0; i < 100; i++)
        gaussian_5x5_rvv(src, blurred_rvv, w, h);
    double r2 = now_ms();
    double rvv_gauss = (r2-r1)/100.0;
    printf("[RVV  ] Gaussian Blur:    %.3f ms  (%.1fx speedup vs scalar)\n",
           rvv_gauss, (t2-t1)/(r2-r1));

    double r3 = now_ms();
    for (int i = 0; i < 100; i++)
        sobel_rvv(blurred_rvv, Gx_rvv, Gy_rvv, w, h);
    double r4 = now_ms();
    double rvv_sobel = (r4-r3)/100.0;
    printf("[RVV  ] Sobel Gradient:   %.3f ms  (%.1fx speedup vs scalar)\n",
           rvv_sobel, (t4-t3)/(r4-r3));

    double r5 = now_ms();
    for (int i = 0; i < 100; i++)
        magnitude_l1_rvv(Gx_rvv, Gy_rvv, mag_rvv, w, h);
    double r6 = now_ms();
    double rvv_mag = (r6-r5)/100.0;
    printf("[RVV  ] Magnitude L1:     %.3f ms  (%.1fx speedup vs scalar)\n",
           rvv_mag, (t6-t5)/(r6-r5));

    printf("[RVV  ] Magnitude L2:     scalar only\n");
    printf("[RVV  ] Direction:        scalar only\n");

    double rvv_total = rvv_gauss + rvv_sobel + rvv_mag
                     + (t8-t7)/100.0 + (t10-t9)/100.0;
    printf("\n[RVV  ] Total pipeline:   %.3f ms  (%.1fx speedup vs scalar)\n",
           rvv_total, scalar_total / rvv_total);

    free(blurred_rvv);
    free(Gx_rvv); free(Gy_rvv);
    free(mag_rvv);
#endif

    // ── Save outputs ─────────────────────────────────────────────────────────

    char path[256];
    snprintf(path, sizeof(path), "%s_blurred.raw", prefix);
    save_raw(path, blurred, w, h);
    snprintf(path, sizeof(path), "%s_mag_l1.raw", prefix);
    save_raw(path, mag_l1, w, h);
    snprintf(path, sizeof(path), "%s_mag_l2.raw", prefix);
    save_raw(path, mag_l2, w, h);
    snprintf(path, sizeof(path), "%s_dir.raw", prefix);
    save_raw(path, dir, w, h);

    free(src); free(blurred);
    free(Gx);  free(Gy);
    free(mag_l1); free(mag_l2); free(dir);

    printf("Done.\n");
    return 0;
}