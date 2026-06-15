#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include "nms.h"
#include "threshold.h"
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

// ── Print usage ──────────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <width> <height> <input.raw> <output_prefix> [low_thresh] [high_thresh]\n", prog);
    fprintf(stderr, "  low_thresh  : weak edge threshold (default: auto = 5%% of max magnitude)\n");
    fprintf(stderr, "  high_thresh : strong edge threshold (default: auto = 15%% of max magnitude)\n");
    fprintf(stderr, "\nPipeline stages:\n");
    fprintf(stderr, "  1. Gaussian Blur    (5x5, scalar + RVV)\n");
    fprintf(stderr, "  2. Sobel Gradient   (Gx, Gy, scalar + RVV)\n");
    fprintf(stderr, "  3. Magnitude L1/L2  (|Gx|+|Gy|, sqrt(Gx^2+Gy^2))\n");
    fprintf(stderr, "  4. Direction        (0/45/90/135 degrees)\n");
    fprintf(stderr, "  5. NMS              (non-maximum suppression)\n");
    fprintf(stderr, "  6. Thresholding     (double threshold + hysteresis)\n");
}

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    const char* in_path = argv[3];
    const char* prefix  = argv[4];

    // Optional threshold overrides
    int user_low_thresh = -1;
    int user_high_thresh = -1;
    if (argc > 5) user_low_thresh  = atoi(argv[5]);
    if (argc > 6) user_high_thresh = atoi(argv[6]);

    uint8_t* src     = load_raw(in_path, w, h);
    uint8_t* blurred = alloc_image(w, h);
    int16_t* Gx      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    int16_t* Gy      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    uint8_t* mag_l1  = alloc_image(w, h);
    uint8_t* mag_l2  = alloc_image(w, h);
    uint8_t* dir     = alloc_image(w, h);
    uint8_t* nms_out = alloc_image(w, h);
    uint8_t* edge_out = alloc_image(w, h);

    printf("=== Canny Edge Detection Pipeline ===\n");
    printf("Image: %dx%d\n", w, h);
    if (user_low_thresh >= 0 && user_high_thresh >= 0)
        printf("Thresholds: low=%d, high=%d (user-specified)\n", user_low_thresh, user_high_thresh);
    else
        printf("Thresholds: auto (low=5%% of max, high=15%% of max)\n");
    printf("\n");

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

    // ── Bonus stages: NMS + Thresholding ────────────────────────────────────

    double t_nms1 = now_ms();
    for (int i = 0; i < 100; i++)
        non_maximum_suppression(mag_l1, dir, nms_out, w, h);
    double t_nms2 = now_ms();
    printf("[Scalar] NMS:              %.3f ms\n", (t_nms2 - t_nms1) / 100.0);

    // Determine thresholds
    uint8_t low_thresh, high_thresh;
    if (user_low_thresh >= 0 && user_high_thresh >= 0) {
        low_thresh  = static_cast<uint8_t>(user_low_thresh);
        high_thresh = static_cast<uint8_t>(user_high_thresh);
    } else {
        // Auto: compute max of L1 magnitude and derive thresholds
        uint8_t max_mag = 0;
        for (int i = 0; i < w * h; ++i)
            if (mag_l1[i] > max_mag) max_mag = mag_l1[i];
        high_thresh = static_cast<uint8_t>(max_mag * 0.15);
        low_thresh  = static_cast<uint8_t>(max_mag * 0.05);
        if (high_thresh < 10) high_thresh = 10;
        if (low_thresh < 5)   low_thresh = 5;
        printf("[Auto]   max_mag=%d  high=%d  low=%d\n", max_mag, high_thresh, low_thresh);
    }

    double t_thresh1 = now_ms();
    for (int i = 0; i < 100; i++)
        threshold_and_hysteresis(mag_l1, edge_out, w, h, low_thresh, high_thresh);
    double t_thresh2 = now_ms();
    printf("[Scalar] Thresholding:     %.3f ms\n", (t_thresh2 - t_thresh1) / 100.0);

    double scalar_total = (t2-t1 + t4-t3 + t6-t5 + t8-t7 + t10-t9 +
                           t_nms2-t_nms1 + t_thresh2-t_thresh1) / 100.0;
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
    printf("[RVV  ] NMS:              scalar only (bonus stage)\n");
    printf("[RVV  ] Thresholding:     scalar only (bonus stage)\n");

    double rvv_total = rvv_gauss + rvv_sobel + rvv_mag
                     + (t8-t7)/100.0 + (t10-t9)/100.0
                     + (t_nms2-t_nms1)/100.0 + (t_thresh2-t_thresh1)/100.0;
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
    snprintf(path, sizeof(path), "%s_nms.raw", prefix);
    save_raw(path, nms_out, w, h);
    snprintf(path, sizeof(path), "%s_edges.raw", prefix);
    save_raw(path, edge_out, w, h);

    printf("\nOutputs saved:\n");
    printf("  %s_blurred.raw\n", prefix);
    printf("  %s_mag_l1.raw, %s_mag_l2.raw\n", prefix, prefix);
    printf("  %s_dir.raw\n", prefix);
    printf("  %s_nms.raw   (non-maximum suppression)\n", prefix);
    printf("  %s_edges.raw (final edges after hysteresis)\n", prefix);
    printf("\nDone.\n");

    free(src); free(blurred);
    free(Gx);  free(Gy);
    free(mag_l1); free(mag_l2); free(dir);
    free(nms_out); free(edge_out);

    return 0;
}
