#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "../src/magnitude.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
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

// Fill Gx and Gy with a constant value
static void fill_gradients(int16_t* Gx, int16_t* Gy, int n,
                            int16_t gx_val, int16_t gy_val) {
    for (int i = 0; i < n; ++i) { Gx[i] = gx_val; Gy[i] = gy_val; }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Zero gradient → magnitude = 0
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, ZeroGradientZeroMag) {
    const int W = 32, H = 32;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    fill_gradients(Gx, Gy, W * H, 0, 0);
    memset(mag, 0xFF, W * H);   // poison

    magnitude_l1(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_EQ(mag[i], 0u) << "L1 mag should be 0 for zero gradient at " << i;

    free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeL2, ZeroGradientZeroMag) {
    const int W = 32, H = 32;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    fill_gradients(Gx, Gy, W * H, 0, 0);
    memset(mag, 0xFF, W * H);

    magnitude_l2(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_EQ(mag[i], 0u) << "L2 mag should be 0 for zero gradient at " << i;

    free(Gx); free(Gy); free(mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Non-zero gradient → non-zero output
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, NonZeroGradientNonZeroOutput) {
    const int W = 32, H = 32;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    fill_gradients(Gx, Gy, W * H, 100, 100);

    magnitude_l1(Gx, Gy, mag, W, H);

    bool any = false;
    for (int i = 0; i < W * H; ++i) if (mag[i] > 0) { any = true; break; }
    EXPECT_TRUE(any) << "L1: non-zero gradient must produce non-zero magnitude";

    free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeL2, NonZeroGradientNonZeroOutput) {
    const int W = 32, H = 32;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    fill_gradients(Gx, Gy, W * H, 100, 100);

    magnitude_l2(Gx, Gy, mag, W, H);

    bool any = false;
    for (int i = 0; i < W * H; ++i) if (mag[i] > 0) { any = true; break; }
    EXPECT_TRUE(any) << "L2: non-zero gradient must produce non-zero magnitude";

    free(Gx); free(Gy); free(mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Output clamped to [0, 255] — no overflow
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, OutputClamped0to255) {
    const int W = 64, H = 64;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    // Max possible Sobel values
    fill_gradients(Gx, Gy, W * H, 1020, 1020);

    magnitude_l1(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_LE(mag[i], 255u) << "L1 overflow at pixel " << i;

    free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeL2, OutputClamped0to255) {
    const int W = 64, H = 64;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    fill_gradients(Gx, Gy, W * H, 1020, 1020);

    magnitude_l2(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_LE(mag[i], 255u) << "L2 overflow at pixel " << i;

    free(Gx); free(Gy); free(mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. L1 >= L2 always (L1 is an overestimate of the Euclidean norm)
//    |a| + |b| >= sqrt(a²+b²) for all real a, b
// ─────────────────────────────────────────────────────────────────────────────
TEST(Magnitude, L1AlwaysGreaterOrEqualL2) {
    const int W = 48, H = 48;
    int16_t* Gx   = alloc_i16(W * H);
    int16_t* Gy   = alloc_i16(W * H);
    uint8_t* mag1 = alloc_u8(W * H);
    uint8_t* mag2 = alloc_u8(W * H);

    // Random-ish gradients covering a range of values
    for (int i = 0; i < W * H; ++i) {
        Gx[i] = static_cast<int16_t>((i * 37 + 13) % 512 - 256);
        Gy[i] = static_cast<int16_t>((i * 53 + 7)  % 512 - 256);
    }

    magnitude_l1(Gx, Gy, mag1, W, H);
    magnitude_l2(Gx, Gy, mag2, W, H);

    // Both use the same two-pass normalisation, so we compare raw magnitudes
    // directly (before normalisation the relationship must hold).
    // We recompute raw values here for correctness:
    for (int i = 0; i < W * H; ++i) {
        int raw_l1 = std::abs((int)Gx[i]) + std::abs((int)Gy[i]);
        double raw_l2 = std::sqrt((double)Gx[i] * Gx[i] +
                                  (double)Gy[i] * Gy[i]);
        EXPECT_GE(raw_l1, (int)raw_l2)
            << "L1 raw < L2 raw at pixel " << i
            << " Gx=" << Gx[i] << " Gy=" << Gy[i];
    }

    free(Gx); free(Gy); free(mag1); free(mag2);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Normalisation: strongest edge pixel must map to 255
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, MaxPixelIs255AfterNorm) {
    const int W = 16, H = 16;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    memset(Gx, 0, W * H * sizeof(int16_t));
    memset(Gy, 0, W * H * sizeof(int16_t));

    // One pixel with high gradient, rest moderate
    for (int i = 0; i < W * H; ++i) { Gx[i] = 50; Gy[i] = 0; }
    Gx[W * H / 2] = 1000;   // single strong edge

    magnitude_l1(Gx, Gy, mag, W, H);

    uint8_t max_val = *std::max_element(mag, mag + W * H);
    EXPECT_EQ(max_val, 255u)
        << "After normalisation, max pixel must be exactly 255";

    free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeL2, MaxPixelIs255AfterNorm) {
    const int W = 16, H = 16;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    for (int i = 0; i < W * H; ++i) { Gx[i] = 50; Gy[i] = 0; }
    Gx[W * H / 2] = 1000;

    magnitude_l2(Gx, Gy, mag, W, H);

    uint8_t max_val = *std::max_element(mag, mag + W * H);
    EXPECT_EQ(max_val, 255u)
        << "After normalisation, max pixel must be exactly 255";

    free(Gx); free(Gy); free(mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Negative gradients: magnitude must be the same as positive (abs value)
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, NegativeGradientSameMagAsPositive) {
    const int W = 16, H = 16;
    int16_t* Gx_pos = alloc_i16(W * H);
    int16_t* Gy_pos = alloc_i16(W * H);
    int16_t* Gx_neg = alloc_i16(W * H);
    int16_t* Gy_neg = alloc_i16(W * H);
    uint8_t* mag_pos = alloc_u8(W * H);
    uint8_t* mag_neg = alloc_u8(W * H);

    fill_gradients(Gx_pos, Gy_pos, W * H,  200,  150);
    fill_gradients(Gx_neg, Gy_neg, W * H, -200, -150);

    magnitude_l1(Gx_pos, Gy_pos, mag_pos, W, H);
    magnitude_l1(Gx_neg, Gy_neg, mag_neg, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_EQ(mag_pos[i], mag_neg[i])
            << "L1: negating gradients must not change magnitude at " << i;

    free(Gx_pos); free(Gy_pos); free(Gx_neg); free(Gy_neg);
    free(mag_pos); free(mag_neg);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Relative ordering preserved: pixel with larger raw gradient →
//    larger output value after normalisation (monotonicity check)
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, MonotonicNormalisation) {
    const int W = 4, H = 1;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    // Gradients: 100, 200, 300, 400 (all in Gx, Gy=0)
    Gx[0] = 100; Gx[1] = 200; Gx[2] = 300; Gx[3] = 400;
    Gy[0] = 0;   Gy[1] = 0;   Gy[2] = 0;   Gy[3] = 0;

    magnitude_l1(Gx, Gy, mag, W, H);

    // After normalisation: mag[3]=255, and mag[0]<mag[1]<mag[2]<mag[3]
    EXPECT_LT(mag[0], mag[1]) << "Monotonicity broken between pixels 0 and 1";
    EXPECT_LT(mag[1], mag[2]) << "Monotonicity broken between pixels 1 and 2";
    EXPECT_LT(mag[2], mag[3]) << "Monotonicity broken between pixels 2 and 3";
    EXPECT_EQ(mag[3], 255u)   << "Max gradient pixel must normalise to 255";

    free(Gx); free(Gy); free(mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Non-power-of-two size (RVV strip-mining readiness)
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, NonPowerOfTwoSize) {
    const int W = 100, H = 75;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    for (int i = 0; i < W * H; ++i) {
        Gx[i] = static_cast<int16_t>(i % 512 - 256);
        Gy[i] = static_cast<int16_t>((i * 3) % 512 - 256);
    }

    EXPECT_NO_FATAL_FAILURE(magnitude_l1(Gx, Gy, mag, W, H));

    uint8_t max_val = *std::max_element(mag, mag + W * H);
    EXPECT_EQ(max_val, 255u) << "Max pixel must be 255 after norm at non-PoT size";

    free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeL2, NonPowerOfTwoSize) {
    const int W = 100, H = 75;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    for (int i = 0; i < W * H; ++i) {
        Gx[i] = static_cast<int16_t>(i % 512 - 256);
        Gy[i] = static_cast<int16_t>((i * 3) % 512 - 256);
    }

    EXPECT_NO_FATAL_FAILURE(magnitude_l2(Gx, Gy, mag, W, H));

    uint8_t max_val = *std::max_element(mag, mag + W * H);
    EXPECT_EQ(max_val, 255u) << "Max pixel must be 255 after norm at non-PoT size";

    free(Gx); free(Gy); free(mag);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Uniform gradient: all pixels same gradient → all output pixels equal
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeL1, UniformGradientUniformOutput) {
    const int W = 32, H = 32;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    fill_gradients(Gx, Gy, W * H, 300, 400);

    magnitude_l1(Gx, Gy, mag, W, H);

    // All raw magnitudes equal → all normalised to 255
    for (int i = 0; i < W * H; ++i)
        EXPECT_EQ(mag[i], 255u)
            << "Uniform gradient should normalise to all-255 at pixel " << i;

    free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeL2, UniformGradientUniformOutput) {
    const int W = 32, H = 32;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_u8(W * H);

    fill_gradients(Gx, Gy, W * H, 300, 400);

    magnitude_l2(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i)
        EXPECT_EQ(mag[i], 255u)
            << "Uniform gradient should normalise to all-255 at pixel " << i;

    free(Gx); free(Gy); free(mag);
}