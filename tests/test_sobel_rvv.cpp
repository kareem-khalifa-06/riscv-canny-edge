
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../src/sobel.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (same conventions as test_sobel.cpp)
// ─────────────────────────────────────────────────────────────────────────────

static uint8_t* alloc_u8(int n) {
    void* p = aligned_alloc(64, ((size_t)n + 63) & ~63);
    EXPECT_NE(p, nullptr);
    return static_cast<uint8_t*>(p);
}

static int16_t* alloc_i16(int n) {
    void* p = aligned_alloc(64, ((size_t)n * sizeof(int16_t) + 63) & ~63);
    EXPECT_NE(p, nullptr);
    return static_cast<int16_t*>(p);
}

// Run both implementations on the same src and assert pixel-exact match.
// Allows ±1 tolerance since the scalar uses int accumulation and RVV uses i16.
static void check_equiv(const char* label, uint8_t* src, int w, int h,
                        int tolerance = 0)
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

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(gx_ref[i], gx_rvv[i], tolerance)
            << label << " — Gx mismatch at pixel " << i
            << " (x=" << i % w << " y=" << i / w << ")"
            << "  scalar=" << gx_ref[i]
            << "  rvv=" << gx_rvv[i];
        EXPECT_NEAR(gy_ref[i], gy_rvv[i], tolerance)
            << label << " — Gy mismatch at pixel " << i
            << " (x=" << i % w << " y=" << i / w << ")"
            << "  scalar=" << gy_ref[i]
            << "  rvv=" << gy_rvv[i];
    }

    free(gx_ref); free(gy_ref);
    free(gx_rvv); free(gy_rvv);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Uniform image → Gx = 0, Gy = 0 everywhere
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, UniformImageZeroGradient) {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    memset(src, 128, N);

    check_equiv("uniform_128", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Sharp vertical edge: large |Gx|, small |Gy|
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, VerticalEdge) {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x < W / 2) ? 0 : 255;

    check_equiv("vertical_edge", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Sharp horizontal edge: large |Gy|, small |Gx|
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, HorizontalEdge) {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (y < H / 2) ? 0 : 255;

    check_equiv("horizontal_edge", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Diagonal edge → both Gx and Gy non-zero
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, DiagonalEdge) {
    const int W = 32, H = 32, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x > y) ? 255 : 0;

    check_equiv("diagonal_edge", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. All-black image → all gradients zero
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, AllBlackZeroGradient) {
    const int W = 64, H = 64, N = W * H;
    uint8_t* src = alloc_u8(N);
    memset(src, 0, N);

    check_equiv("all_black", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. All-white image → all gradients zero (uniform)
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, AllWhiteZeroGradient) {
    const int W = 64, H = 64, N = W * H;
    uint8_t* src = alloc_u8(N);
    memset(src, 255, N);

    check_equiv("all_white", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Checkerboard — maximises local contrast, exercises all kernel taps
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, CheckerboardMaxContrast) {
    const int W = 64, H = 64, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = ((x + y) & 1) ? 255 : 0;

    check_equiv("checkerboard", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Non-power-of-two width — exercises RVV tail/strip-mining
//     W=100 is not a multiple of any typical VLMAX (8, 16, 32, 64).
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, NonPowerOfTwoWidth) {
    const int W = 100, H = 75, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 7 + 3) % 256);

    check_equiv("non_pot_100x75", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Single pixel wide — W=1, H=127
//     Every interior pixel needs left/right boundary handling.
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, SingleColumn) {
    const int W = 1, H = 127, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 13) % 256);

    check_equiv("single_column_1x127", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Large image stress test — 1920×1080
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, LargeImage1080p) {
    const int W = 1920, H = 1080, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 17 + 5) % 256);

    check_equiv("large_1920x1080", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Random-ish gradients — general correctness across wide value range
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, RandomImage) {
    const int W = 48, H = 48, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 37 + 13) % 256);

    check_equiv("random_48x48", src, W, H);

    free(src);
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Small 3×3 image — minimal size, every pixel is a border pixel
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, MinimalSize3x3) {
    const int W = 3, H = 3, N = W * H;
    uint8_t src[9] = {
        0,   0,   0,
        0, 255, 255,
        0, 255, 255
    };

    check_equiv("minimal_3x3", src, W, H);
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Width = 3 (minimum for 1 interior column)
// ─────────────────────────────────────────────────────────────────────────────
TEST(SobelRVV, WidthThree) {
    const int W = 3, H = 16, N = W * H;
    uint8_t* src = alloc_u8(N);
    for (int i = 0; i < N; ++i)
        src[i] = static_cast<uint8_t>((i * 11) % 256);

    check_equiv("width_3", src, W, H);

    free(src);
}
