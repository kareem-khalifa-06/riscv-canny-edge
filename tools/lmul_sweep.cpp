
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <algorithm>

// Declare the two LMUL variants (defined in rvv/gaussian_rvv.cpp)
extern "C" {
void gaussian_5x5_rvv_m1(const uint8_t* src, uint8_t* dst, int w, int h);
void gaussian_5x5_rvv_m2(const uint8_t* src, uint8_t* dst, int w, int h);
}

static double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static void fill_random(uint8_t* buf, int n) {
    for (int i = 0; i < n; ++i) {
        buf[i] = static_cast<uint8_t>(rand() & 0xFF);
    }
}

static bool compare_buffers(const uint8_t* a, const uint8_t* b, int n, int& first_diff) {
    for (int i = 0; i < n; ++i) {
        // Allow ±1 tolerance for fixed-point rounding differences
        int diff = abs((int)a[i] - (int)b[i]);
        if (diff > 1) {
            first_diff = i;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const int W = (argc > 1) ? atoi(argv[1]) : 512;
    const int H = (argc > 2) ? atoi(argv[2]) : 512;
    const int N = W * H;
    const int ITERATIONS = 100;

    printf("=== LMUL Sweep: Gaussian 5x5 ===\n");
    printf("Image: %dx%d (%d pixels)\n", W, H, N);
    printf("Iterations: %d\n\n", ITERATIONS);

    // Allocate aligned buffers
    uint8_t* src = (uint8_t*)aligned_alloc(64, N);
    uint8_t* dst_m1 = (uint8_t*)aligned_alloc(64, N);
    uint8_t* dst_m2 = (uint8_t*)aligned_alloc(64, N);

    srand(42);
    fill_random(src, N);

    // Warm-up run (ensure caches are warm)
    gaussian_5x5_rvv_m1(src, dst_m1, W, H);
    gaussian_5x5_rvv_m2(src, dst_m2, W, H);

    // Verify correctness: m1 vs m2 should match within ±1
    int first_diff = -1;
    bool match = compare_buffers(dst_m1, dst_m2, N, first_diff);
    if (!match) {
        printf("⚠️  WARNING: m1 and m2 outputs differ at pixel %d!\n", first_diff);
        printf("    m1[%d] = %d, m2[%d] = %d\n",
               first_diff, dst_m1[first_diff],
               first_diff, dst_m2[first_diff]);
    } else {
        printf("✅ m1 and m2 outputs match within ±1 tolerance\n\n");
    }

    // Time LMUL=1
    double t1 = now_ms();
    for (int i = 0; i < ITERATIONS; ++i) {
        gaussian_5x5_rvv_m1(src, dst_m1, W, H);
    }
    double t2 = now_ms();
    double m1_ms = (t2 - t1) / ITERATIONS;

    // Time LMUL=2
    double t3 = now_ms();
    for (int i = 0; i < ITERATIONS; ++i) {
        gaussian_5x5_rvv_m2(src, dst_m2, W, H);
    }
    double t4 = now_ms();
    double m2_ms = (t4 - t3) / ITERATIONS;

    // Results
    printf("Results (average over %d iterations):\n", ITERATIONS);
    printf("  LMUL=1 (m1):  %.4f ms\n", m1_ms);
    printf("  LMUL=2 (m2):  %.4f ms\n", m2_ms);

    if (m2_ms < m1_ms) {
        double speedup = m1_ms / m2_ms;
        printf("\n  LMUL=2 is %.2fx faster than LMUL=1\n", speedup);
        printf("  → Recommendation: Use LMUL=2 for Gaussian\n");
    } else if (m1_ms < m2_ms) {
        double speedup = m2_ms / m1_ms;
        printf("\n  LMUL=1 is %.2fx faster than LMUL=2\n", speedup);
        printf("  → Recommendation: Use LMUL=1 for Gaussian\n");
    } else {
        printf("\n  LMUL=1 and LMUL=2 are equivalent in performance\n");
        printf("  → Recommendation: Use LMUL=1 (lower register pressure)\n");
    }

    free(src);
    free(dst_m1);
    free(dst_m2);

    return 0;
}
