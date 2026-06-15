// ─────────────────────────────────────────────────────────────────────────────
// QEMU-side Gaussian RVV Equivalence Test
//
// Runs the scalar and RVV Gaussian implementations on the same input and
// verifies outputs match within ±1 (fixed-point rounding tolerance).
//
// Build:  make $(RV_DIR)/Test_gaussian_rvv_qemu
// Run:    qemu-riscv64 -cpu rv64,v=true,vlen=256 $(RV_DIR)/Test_gaussian_rvv_qemu
// ─────────────────────────────────────────────────────────────────────────────

#include "../src/gaussian.h"
#include "../src/image_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

// Declared in rvv/gaussian_rvv.cpp
extern "C" void gaussian_5x5_rvv(const uint8_t* src, uint8_t* dst, int w, int h);

static int tests_passed = 0;
static int tests_failed = 0;

static void expect_near(const char* name,
                        const uint8_t* a, const uint8_t* b,
                        int n, int tol)
{
    int first_diff = -1;
    int max_diff = 0;
    for (int i = 0; i < n; ++i) {
        int d = std::abs((int)a[i] - (int)b[i]);
        if (d > max_diff) max_diff = d;
        if (d > tol && first_diff < 0) first_diff = i;
    }
    if (first_diff < 0) {
        printf("  PASS: %s (max_diff=%d <= tol=%d)\n", name, max_diff, tol);
        tests_passed++;
    } else {
        printf("  FAIL: %s — first diff at pixel %d "
               "(scalar=%d, rvv=%d, diff=%d > tol=%d)\n",
               name, first_diff, a[first_diff], b[first_diff],
               std::abs((int)a[first_diff] - (int)b[first_diff]), tol);
        tests_failed++;
    }
}

static void fill_uniform(uint8_t* buf, int n, uint8_t val) {
    for (int i = 0; i < n; ++i) buf[i] = val;
}

static void fill_random(uint8_t* buf, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) buf[i] = (uint8_t)(rand() & 0xFF);
}

static void impulse(uint8_t* buf, int w, int h, int cx, int cy) {
    memset(buf, 0, w * h);
    if (cx >= 0 && cx < w && cy >= 0 && cy < h)
        buf[cy * w + cx] = 255;
}

static void rect(uint8_t* buf, int w, int h,
                 int x1, int y1, int x2, int y2,
                 uint8_t fg, uint8_t bg)
{
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            buf[y * w + x] = (x >= x1 && x < x2 && y >= y1 && y < y2) ? fg : bg;
}

int main() {
    printf("=== Gaussian RVV Equivalence Tests (QEMU) ===\n\n");

    // Test 1: Uniform 128 (non-power-of-2 to stress strip-mining)
    {
        const int W = 47, H = 53, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        fill_uniform(src, N, 128);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("uniform128_47x53", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 2: All black
    {
        const int W = 64, H = 64, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        fill_uniform(src, N, 0);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("all_black_64x64", dst_scalar, dst_rvv, N, 0);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 3: Impulse at center
    {
        const int W = 64, H = 64, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        impulse(src, W, H, 32, 32);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("impulse_center_64x64", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 4: White rectangle on black
    {
        const int W = 100, H = 100, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        rect(src, W, H, 25, 25, 75, 75, 255, 0);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("white_rect_100x100", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 5: Random data (non-power-of-2)
    {
        const int W = 101, H = 97, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        fill_random(src, N, 12345);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("random_101x97", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 6: Edge case — minimum viable image (5x5, kernel exactly fits)
    {
        const int W = 5, H = 5, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        fill_random(src, N, 99);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("min_size_5x5", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 7: 1920x1080 stress test
    {
        const int W = 1920, H = 1080, N = W * H;
        printf("  Running 1920x1080 stress test ( allocating %d MB )...\n",
               (int)(N * 3 / (1024 * 1024)));
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        fill_random(src, N, 42);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("stress_1920x1080", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 8: All white
    {
        const int W = 48, H = 48, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        fill_uniform(src, N, 255);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("all_white_48x48", dst_scalar, dst_rvv, N, 0);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 9: Impulse at corner (stresses boundary handling)
    {
        const int W = 64, H = 64, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        impulse(src, W, H, 0, 0);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("impulse_corner_64x64", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    // Test 10: Impulse at opposite corner
    {
        const int W = 64, H = 64, N = W * H;
        uint8_t* src = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_scalar = (uint8_t*)aligned_alloc(64, N);
        uint8_t* dst_rvv = (uint8_t*)aligned_alloc(64, N);
        impulse(src, W, H, 63, 63);
        gaussian_5x5(src, dst_scalar, W, H);
        gaussian_5x5_rvv(src, dst_rvv, W, H);
        expect_near("impulse_far_corner_64x64", dst_scalar, dst_rvv, N, 1);
        free(src); free(dst_scalar); free(dst_rvv);
    }

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
