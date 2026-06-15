
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../src/sobel.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static uint8_t* alloc_u8(int n) {
    void* p = aligned_alloc(64, ((size_t)n + 63) & ~63);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return static_cast<uint8_t*>(p);
}

static int16_t* alloc_i16(int n) {
    void* p = aligned_alloc(64, ((size_t)n * sizeof(int16_t) + 63) & ~63);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return static_cast<int16_t*>(p);
}

// Compare scalar vs RVV Sobel outputs.  Returns number of mismatches.
static int compare(const char* label, uint8_t* src, int w, int h)
{
    int n = w * h;
    int16_t* gx_ref = alloc_i16(n);
    int16_t* gy_ref = alloc_i16(n);
    int16_t* gx_rvv = alloc_i16(n);
    int16_t* gy_rvv = alloc_i16(n);

    memset(gx_ref, 0xAA, n * sizeof(int16_t));
    memset(gy_ref, 0xAA, n * sizeof(int16_t));
    memset(gx_rvv, 0x55, n * sizeof(int16_t));
    memset(gy_rvv, 0x55, n * sizeof(int16_t));

    sobel    (src, gx_ref, gy_ref, w, h);
    sobel_rvv(src, gx_rvv, gy_rvv, w, h);

    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (gx_ref[i] != gx_rvv[i] || gy_ref[i] != gy_rvv[i]) {
            if (mismatches < 8) {
                fprintf(stderr,
                    "  [%s] MISMATCH pixel %d (x=%d y=%d): "
                    "scalar_Gx=%d rvv_Gx=%d  scalar_Gy=%d rvv_Gy=%d\n",
                    label, i, i % w, i / w,
                    gx_ref[i], gx_rvv[i], gy_ref[i], gy_rvv[i]);
            }
            ++mismatches;
        }
    }

    if (mismatches == 0)
        printf("PASS  %-40s  %dx%d\n", label, w, h);
    else
        printf("FAIL  %-40s  %dx%d  (%d mismatches)\n",
               label, w, h, mismatches);

    free(gx_ref); free(gy_ref);
    free(gx_rvv); free(gy_rvv);
    return mismatches;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test cases
// ─────────────────────────────────────────────────────────────────────────────

// T1: Uniform image → zero gradients
static int t_uniform() {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    memset(src, 128, N);
    int r = compare("uniform_128", src, W, H);
    free(src);
    return r;
}

// T2: Vertical edge → large Gx, small Gy
static int t_vertical_edge() {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x < W / 2) ? 0 : 255;
    int r = compare("vertical_edge", src, W, H);
    free(src);
    return r;
}

// T3: Horizontal edge → large Gy, small Gx
static int t_horizontal_edge() {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (y < H / 2) ? 0 : 255;
    int r = compare("horizontal_edge", src, W, H);
    free(src);
    return r;
}

// T4: Diagonal edge → both non-zero
static int t_diagonal_edge() {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x > y) ? 255 : 0;
    int r = compare("diagonal_edge", src, W, H);
    free(src);
    return r;
}

// T5: All black → all zero
static int t_all_black() {
    const int W = 64, H = 64, N = W * H;
    uint8_t* src = alloc_u8(N);
    memset(src, 0, N);
    int r = compare("all_black", src, W, H);
    free(src);
    return r;
}

// T6: Checkerboard — max contrast
static int t_checkerboard() {
    const int W = 64, H = 64, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = ((x + y) & 1) ? 255 : 0;
    int r = compare("checkerboard", src, W, H);
    free(src);
    return r;
}

// T7: Non-power-of-two — exercises tail loop
static int t_non_pot() {
    const int W = 100, H = 75, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
    int r = compare("non_pot_100x75", src, W, H);
    free(src);
    return r;
}

// T8: Single column — W=1
static int t_single_column() {
    const int W = 1, H = 127, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 13) % 256);
    int r = compare("single_column_1x127", src, W, H);
    free(src);
    return r;
}

// T9: Large image stress test
static int t_large() {
    const int W = 1920, H = 1080, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 17 + 5) % 256);
    int r = compare("large_1920x1080", src, W, H);
    free(src);
    return r;
}

// T10: Small 3×3 — minimal size
static int t_minimal() {
    uint8_t src[9] = {
        0,   0,   0,
        0, 255, 255,
        0, 255, 255
    };
    return compare("minimal_3x3", src, 3, 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== sobel_rvv equivalence tests (scalar vs RVV) ===\n\n");

    int total_fail = 0;

    total_fail += t_uniform();
    total_fail += t_vertical_edge();
    total_fail += t_horizontal_edge();
    total_fail += t_diagonal_edge();
    total_fail += t_all_black();
    total_fail += t_checkerboard();
    total_fail += t_non_pot();
    total_fail += t_single_column();
    total_fail += t_large();
    total_fail += t_minimal();

    printf("\n");
    if (total_fail == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("FAILED — %d total pixel mismatches across all tests\n", total_fail);

    return total_fail ? 1 : 0;
}
