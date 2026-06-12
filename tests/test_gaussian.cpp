#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../src/gaussian.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static uint8_t* alloc64(int n) {
    void* p = aligned_alloc(64, ((size_t)n + 63) & ~63);
    EXPECT_NE(p, nullptr);
    return static_cast<uint8_t*>(p);
}

static void fill(uint8_t* buf, int n, uint8_t val) {
    memset(buf, val, n);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Uniform image: any constant input must emerge constant (interior ±1)
//    Zero-padding causes border pixels to bleed toward 0, so we only
//    check the interior (2-pixel margin from every edge).
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, UniformGrey128InteriorPreserved) {
    const int W = 32, H = 32;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);
    fill(src, W * H, 128);
    fill(dst, W * H, 0);

    gaussian_5x5(src, dst, W, H);

    for (int y = 2; y < H - 2; ++y)
        for (int x = 2; x < W - 2; ++x)
            EXPECT_NEAR((int)dst[y * W + x], 128, 1)
                << "Failed at (" << x << "," << y << ")";

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. All-black image → output must be all-black (zero × anything = 0)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, AllBlackStaysBlack) {
    const int W = 64, H = 64;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);
    fill(src, W * H, 0);
    fill(dst, W * H, 255);   // poison output

    gaussian_5x5(src, dst, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_EQ(dst[i], 0u) << "Non-zero output at pixel " << i;

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. All-white image interior: should stay 255 (kernel is normalised to 1)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, AllWhiteInteriorStays255) {
    const int W = 32, H = 32;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);
    fill(src, W * H, 255);
    fill(dst, W * H, 0);

    gaussian_5x5(src, dst, W, H);

    // Interior (2-pixel margin): kernel fully inside → no zero-pad contribution
    for (int y = 2; y < H - 2; ++y)
        for (int x = 2; x < W - 2; ++x)
            EXPECT_EQ(dst[y * W + x], 255u)
                << "Interior pixel dimmed at (" << x << "," << y << ")";

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Impulse response: single bright pixel must spread symmetrically
//    Place impulse at exact centre. The 5×5 kernel coefficients are symmetric,
//    so dst[cy-1][cx] == dst[cy+1][cx] and dst[cy][cx-1] == dst[cy][cx+1].
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, ImpulseSpreadsSyimmetrically) {
    const int W = 32, H = 32;
    const int cx = W / 2, cy = H / 2;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);
    fill(src, W * H, 0);
    src[cy * W + cx] = 255;
    fill(dst, W * H, 0);

    gaussian_5x5(src, dst, W, H);

    // Centre must be the brightest pixel in the output
    uint8_t centre = dst[cy * W + cx];
    EXPECT_GT(centre, 0u) << "Centre should be non-zero after blur";

    // Left/right symmetry
    EXPECT_EQ(dst[cy * W + (cx - 1)], dst[cy * W + (cx + 1)])
        << "Left/right asymmetry";
    // Up/down symmetry
    EXPECT_EQ(dst[(cy - 1) * W + cx], dst[(cy + 1) * W + cx])
        << "Up/down asymmetry";
    // Diagonal symmetry
    EXPECT_EQ(dst[(cy - 1) * W + (cx - 1)], dst[(cy + 1) * W + (cx + 1)])
        << "Diagonal asymmetry";

    // Pixels at distance >2 from impulse must be 0 (kernel radius = 2)
    EXPECT_EQ(dst[(cy - 3) * W + cx], 0u) << "Kernel leaked beyond radius 2";
    EXPECT_EQ(dst[cy * W + (cx + 3)], 0u) << "Kernel leaked beyond radius 2";

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Zero-padding: top-left corner output must be ≥ 0 and < input value
//    With zero-pad the corner pixel averages with zeros, so output < input.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, ZeroPaddingCornerBleach) {
    const int W = 32, H = 32;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);
    fill(src, W * H, 200);
    fill(dst, W * H, 0);

    gaussian_5x5(src, dst, W, H);

    // Corner pixel (0,0): many kernel taps fall outside → zeros dilute output
    EXPECT_GE(dst[0], 0u);
    EXPECT_LT(dst[0], 200u) << "Corner should be dimmer than interior with zero-padding";

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Output values must never exceed 255 (no overflow / undefined behaviour)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, OutputClampedTo255) {
    const int W = 64, H = 64;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);
    fill(src, W * H, 255);
    fill(dst, W * H, 0);

    gaussian_5x5(src, dst, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_LE(dst[i], 255u) << "Overflow at pixel " << i;

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Non-power-of-two size: exercise any strip-mining tail case
//    (critical for later RVV equivalence; scalar must handle odd widths)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, NonPowerOfTwoSize) {
    const int W = 100, H = 75;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);

    // Fill with a gradient so every pixel is distinct
    for (int i = 0; i < W * H; ++i)
        src[i] = static_cast<uint8_t>(i & 0xFF);
    fill(dst, W * H, 0);

    // Must not crash or produce garbage
    EXPECT_NO_FATAL_FAILURE(gaussian_5x5(src, dst, W, H));

    // Sanity: output should differ from all-zero
    bool any_nonzero = false;
    for (int i = 0; i < W * H; ++i)
        if (dst[i]) { any_nonzero = true; break; }
    EXPECT_TRUE(any_nonzero) << "All output zeros on gradient image — something is wrong";

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Blur reduces variance: a noisy image should have lower std-dev after blur
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, BlurReducesVariance) {
    const int W = 64, H = 64;
    uint8_t* src = alloc64(W * H);
    uint8_t* dst = alloc64(W * H);

    // Checkerboard: maximum high-frequency content
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = ((x + y) & 1) ? 255 : 0;

    gaussian_5x5(src, dst, W, H);

    // Compute variance for src and dst over interior
    double mean_src = 0, mean_dst = 0;
    int n = 0;
    for (int y = 2; y < H - 2; ++y)
        for (int x = 2; x < W - 2; ++x, ++n) {
            mean_src += src[y * W + x];
            mean_dst += dst[y * W + x];
        }
    mean_src /= n; mean_dst /= n;

    double var_src = 0, var_dst = 0;
    for (int y = 2; y < H - 2; ++y)
        for (int x = 2; x < W - 2; ++x) {
            double ds = src[y * W + x] - mean_src;
            double dd = dst[y * W + x] - mean_dst;
            var_src += ds * ds;
            var_dst += dd * dd;
        }

    EXPECT_LT(var_dst, var_src)
        << "Blur must reduce variance; output is noisier than input";

    free(src); free(dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Template instantiation smoke test: int32_t pixels (for coverage)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Gaussian, TemplateWorksWithInt32) {
    const int W = 16, H = 16;
    // Allocate int32 arrays
    int32_t* src32 = static_cast<int32_t*>(aligned_alloc(64, W * H * sizeof(int32_t)));
    int32_t* dst32 = static_cast<int32_t*>(aligned_alloc(64, W * H * sizeof(int32_t)));

    for (int i = 0; i < W * H; ++i) src32[i] = 100;
    memset(dst32, 0, W * H * sizeof(int32_t));

    gaussian_5x5<int32_t, int64_t, int32_t>(src32, dst32, W, H);

    // Interior should be ≈100
    for (int y = 2; y < H - 2; ++y)
        for (int x = 2; x < W - 2; ++x)
            EXPECT_NEAR(dst32[y * W + x], 100, 1)
                << "Template int32 failed at (" << x << "," << y << ")";

    free(src32); free(dst32);
}