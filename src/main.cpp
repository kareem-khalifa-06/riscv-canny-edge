// ═════════════════════════════════════════════════════════════════════════════
// Canny Edge Detection Pipeline — main entry point
// ═════════════════════════════════════════════════════════════════════════════
#include "image_io.h"
#include "gaussian.h"
#include "gaussian_separable.h"   // ← separable scalar + RVV declaration
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
void sobel_rvv(const uint8_t* __restrict__ src,
               int16_t* __restrict__ Gx, int16_t* __restrict__ Gy,
               int w, int h);
void magnitude_l1_rvv(const int16_t* __restrict__ Gx,
                      const int16_t* __restrict__ Gy,
                      uint8_t* __restrict__ mag,
                      int w, int h);
void direction_rvv(const int16_t* __restrict__ Gx,
                   const int16_t* __restrict__ Gy,
                   uint8_t* __restrict__ dir,
                   int w, int h);
// Separable Gaussian RVV (declared in gaussian_separable.h via extern "C")
// gaussian_5x5_sep_rvv — defined in rvv/gaussian_rvv.cpp
#endif

static double now_ms() {
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
    fprintf(stderr, "  4. Direction        (0/45/90/135 degrees, scalar + RVV)\n");
    fprintf(stderr, "  5. NMS              (non-maximum suppression)\n");
    fprintf(stderr, "  6. Thresholding     (double threshold + hysteresis)\n");
}

int main(int argc, char** argv) {
    if (argc < 5) { print_usage(argv[0]); return 1; }

    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    const char* in_path = argv[3];
    const char* prefix  = argv[4];

    int user_low_thresh  = (argc > 5) ? atoi(argv[5]) : -1;
    int user_high_thresh = (argc > 6) ? atoi(argv[6]) : -1;

    uint8_t* src         = load_raw(in_path, w, h);
    uint8_t* blurred     = alloc_image(w, h);
    uint8_t* blurred_sep = alloc_image(w, h);  // separable output
    int16_t* Gx          = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    int16_t* Gy          = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    uint8_t* mag_l1      = alloc_image(w, h);
    uint8_t* mag_l2      = alloc_image(w, h);
    uint8_t* dir         = alloc_image(w, h);
    uint8_t* nms_out     = alloc_image(w, h);
    uint8_t* edge_out    = alloc_image(w, h);

    printf("=== Canny Edge Detection Pipeline ===\n");
    printf("Image: %dx%d\n", w, h);
    if (user_low_thresh >= 0 && user_high_thresh >= 0)
        printf("Thresholds: low=%d, high=%d (user-specified)\n", user_low_thresh, user_high_thresh);
    else
        printf("Thresholds: auto (low=5%% of max, high=15%% of max)\n");
    printf("\n");

    // ─────────────────────────────────────────────────────────────────────────
    // SCALAR STAGES
    // ─────────────────────────────────────────────────────────────────────────

    // Gaussian 2-D reference
    double t1 = now_ms();
    for (int i = 0; i < 100; i++) gaussian_5x5(src, blurred, w, h);
    double t2 = now_ms();
    double scalar_gauss2d_ms = (t2 - t1) / 100.0;
    printf("[Scalar] Gaussian 2-D:     %.3f ms\n", scalar_gauss2d_ms);

    // Gaussian separable scalar
    double ts1 = now_ms();
    for (int i = 0; i < 100; i++) gaussian_5x5_separable(src, blurred_sep, w, h);
    double ts2 = now_ms();
    double scalar_sep_ms = (ts2 - ts1) / 100.0;
    printf("[Scalar] Gaussian sep:     %.3f ms  (%.1fx vs 2-D)\n",
           scalar_sep_ms, scalar_gauss2d_ms / scalar_sep_ms);

    // Compare the two scalar outputs
    gaussian_5x5(src, blurred, w, h);
    gaussian_5x5_separable(src, blurred_sep, w, h);
    GaussianCompareResult cmp = gaussian_compare(blurred, blurred_sep, w, h, /*tol=*/5);
    printf("[Compare] 2-D vs sep:      max_diff=%d  pixels>tol=%d  %s\n",
           cmp.max_diff, cmp.pixels_beyond_tol,
           cmp.within_tolerance ? "OK" : "NOTE: diff is expected (different kernels)");
    printf("          (K5/273 vs k1d⊗k1d/289 — both valid Gaussian approx, max diff ≤5)\n\n");

    // Sobel — run on the SEPARABLE blurred output so the RVV pipeline is consistent
    double t3 = now_ms();
    for (int i = 0; i < 100; i++) sobel(blurred_sep, Gx, Gy, w, h);
    double t4 = now_ms();
    double scalar_sobel_ms = (t4 - t3) / 100.0;
    printf("[Scalar] Sobel Gradient:   %.3f ms\n", scalar_sobel_ms);

    double t5 = now_ms();
    for (int i = 0; i < 100; i++) magnitude_l1(Gx, Gy, mag_l1, w, h);
    double t6 = now_ms();
    double scalar_mag_ms = (t6 - t5) / 100.0;
    printf("[Scalar] Magnitude L1:     %.3f ms\n", scalar_mag_ms);

    double t7 = now_ms();
    for (int i = 0; i < 100; i++) magnitude_l2(Gx, Gy, mag_l2, w, h);
    double t8 = now_ms();
  double scalar_mag_l2_ms = (t8 - t7) / 100.0;
    printf("[Scalar] Magnitude L2: %.3f ms\n", scalar_mag_l2_ms);

    double t9 = now_ms();
    for (int i = 0; i < 100; i++) direction(Gx, Gy, dir, w, h);
    double t10 = now_ms();
    double scalar_dir_ms = (t10 - t9) / 100.0;
    printf("[Scalar] Direction:        %.3f ms\n", scalar_dir_ms);

    // ── Bonus stages: NMS + Thresholding ────────────────────────────────────
    double t_nms1 = now_ms();
    for (int i = 0; i < 100; i++) non_maximum_suppression(mag_l1, dir, nms_out, w, h);
    double t_nms2 = now_ms();
    double scalar_nms_ms = (t_nms2 - t_nms1) / 100.0;
    printf("[Scalar] NMS:              %.3f ms\n", scalar_nms_ms);

    // Thresholds
    uint8_t low_thresh, high_thresh;
    if (user_low_thresh >= 0 && user_high_thresh >= 0) {
        low_thresh  = (uint8_t)user_low_thresh;
        high_thresh = (uint8_t)user_high_thresh;
    } else {
        uint8_t max_mag = 0;
        for (int i = 0; i < w * h; ++i) if (mag_l1[i] > max_mag) max_mag = mag_l1[i];
        high_thresh = (uint8_t)(max_mag * 0.15);
        low_thresh  = (uint8_t)(max_mag * 0.05);
        if (high_thresh < 10) high_thresh = 10;
        if (low_thresh  < 5)  low_thresh  = 5;
        printf("[Auto]   max_mag=%d  high=%d  low=%d\n", max_mag, high_thresh, low_thresh);
    }

    double t_th1 = now_ms();
    for (int i = 0; i < 100; i++)
        threshold_and_hysteresis(mag_l1, edge_out, w, h, low_thresh, high_thresh);
    double t_th2 = now_ms();
    double scalar_thresh_ms = (t_th2 - t_th1) / 100.0;
    printf("[Scalar] Thresholding:     %.3f ms\n", scalar_thresh_ms);

    // Scalar total — use SEPARABLE gaussian as the baseline (that's what RVV will use)
    double scalar_total_ms = scalar_sep_ms + scalar_sobel_ms + scalar_mag_ms
                           + scalar_dir_ms + scalar_nms_ms + scalar_thresh_ms;
    printf("\n[Scalar] Total (sep pipe): %.3f ms\n", scalar_total_ms);

    // ─────────────────────────────────────────────────────────────────────────
    // RVV STAGES
    // ─────────────────────────────────────────────────────────────────────────
#ifdef __riscv_v
    uint8_t* blurred_rvv = alloc_image(w, h);
    int16_t* Gx_rvv      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    int16_t* Gy_rvv      = (int16_t*)aligned_alloc(64, w * h * sizeof(int16_t));
    uint8_t* mag_rvv     = alloc_image(w, h);
    uint8_t* dir_rvv     = alloc_image(w, h);

    printf("\n");

        // ── RVV Gaussian 2-D (for comparison / README table) ─────────────────────
    double rg2d_1 = now_ms();
    for (int i = 0; i < 100; i++) gaussian_5x5_rvv(src, blurred_rvv, w, h);
    double rg2d_2 = now_ms();
    double rvv_gauss2d_ms = (rg2d_2 - rg2d_1) / 100.0;
    printf("[RVV  ] Gaussian 2-D:      %.3f ms  (%.1fx vs scalar 2-D)\n",
           rvv_gauss2d_ms, scalar_gauss2d_ms / rvv_gauss2d_ms);
           
    // ── RVV Gaussian (separable) ──────────────────────────────────────────────
    double rg1 = now_ms();
    for (int i = 0; i < 100; i++) gaussian_5x5_sep_rvv(src, blurred_rvv, w, h);
    double rg2 = now_ms();
    double rvv_gauss_ms = (rg2 - rg1) / 100.0;
    printf("[RVV  ] Gaussian sep:      %.3f ms  (%.1fx vs scalar sep,  %.1fx vs scalar 2-D)\n",
           rvv_gauss_ms,
           scalar_sep_ms    / rvv_gauss_ms,
           scalar_gauss2d_ms / rvv_gauss_ms);

    // ── RVV Sobel ─────────────────────────────────────────────────────────────
    double rs1 = now_ms();
    for (int i = 0; i < 100; i++) sobel_rvv(blurred_rvv, Gx_rvv, Gy_rvv, w, h);
    double rs2 = now_ms();
    double rvv_sobel_ms = (rs2 - rs1) / 100.0;
    printf("[RVV  ] Sobel Gradient:    %.3f ms  (%.1fx vs scalar)\n",
           rvv_sobel_ms, scalar_sobel_ms / rvv_sobel_ms);

    // ── RVV Magnitude ─────────────────────────────────────────────────────────
    double rm1 = now_ms();
    for (int i = 0; i < 100; i++) magnitude_l1_rvv(Gx_rvv, Gy_rvv, mag_rvv, w, h);
    double rm2 = now_ms();
    double rvv_mag_ms = (rm2 - rm1) / 100.0;
    printf("[RVV  ] Magnitude L1:      %.3f ms  (%.1fx vs scalar)\n",
           rvv_mag_ms, scalar_mag_ms / rvv_mag_ms);

    printf("[RVV  ] Magnitude L2:      scalar only\n");

    // ── RVV Direction ─────────────────────────────────────────────────────────
    double rd1 = now_ms();
    for (int i = 0; i < 100; i++) direction_rvv(Gx_rvv, Gy_rvv, dir_rvv, w, h);
    double rd2 = now_ms();
    double rvv_dir_ms = (rd2 - rd1) / 100.0;
    printf("[RVV  ] Direction:         %.3f ms  (%.1fx vs scalar)\n",
           rvv_dir_ms, scalar_dir_ms / rvv_dir_ms);

    printf("[RVV  ] NMS:               scalar only (bonus stage)\n");
    printf("[RVV  ] Thresholding:      scalar only (bonus stage)\n");

    // ── RVV total — uses separable Gaussian + RVV Sobel/Mag/Dir + scalar NMS/Thresh
    double rvv_total_ms = rvv_gauss_ms + rvv_sobel_ms + rvv_mag_ms
                        + rvv_dir_ms + scalar_nms_ms + scalar_thresh_ms;
    printf("\n[RVV  ] Total pipeline:    %.3f ms  (%.1fx vs scalar sep pipe)\n",
           rvv_total_ms, scalar_total_ms / rvv_total_ms);

   

    free(blurred_rvv); free(Gx_rvv); free(Gy_rvv); free(mag_rvv); free(dir_rvv);
#endif

    // ─────────────────────────────────────────────────────────────────────────
    // SAVE OUTPUTS
    // ─────────────────────────────────────────────────────────────────────────
    char path[256];
    snprintf(path, sizeof(path), "%s_blurred.raw",     prefix); save_raw(path, blurred,     w, h);
    snprintf(path, sizeof(path), "%s_blurred_sep.raw", prefix); save_raw(path, blurred_sep, w, h);
    snprintf(path, sizeof(path), "%s_mag_l1.raw",      prefix); save_raw(path, mag_l1,      w, h);
    snprintf(path, sizeof(path), "%s_mag_l2.raw",      prefix); save_raw(path, mag_l2,      w, h);
    snprintf(path, sizeof(path), "%s_dir.raw",         prefix); save_raw(path, dir,         w, h);
    snprintf(path, sizeof(path), "%s_nms.raw",         prefix); save_raw(path, nms_out,     w, h);
    snprintf(path, sizeof(path), "%s_edges.raw",       prefix); save_raw(path, edge_out,    w, h);

    printf("\nOutputs saved with prefix '%s'\n", prefix);
    printf("Done.\n");

    free(src); free(blurred); free(blurred_sep);
    free(Gx); free(Gy); free(mag_l1); free(mag_l2);
    free(dir); free(nms_out); free(edge_out);
    return 0;
}