
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "../src/magnitude.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (same conventions as test_magnitude.cpp)
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

// Run both implementations on the same (Gx, Gy) and assert pixel-exact match.
static void check_equiv(const char* label,
                         int16_t* Gx, int16_t* Gy,
                         int w, int h)
{
    int n = w * h;
    uint8_t* ref = alloc_u8(n);
    uint8_t* rvv = alloc_u8(n);

    memset(ref, 0xAA, n);   // poison
    memset(rvv, 0x55, n);

    magnitude_l1    (Gx, Gy, ref, w, h);
    magnitude_l1_rvv(Gx, Gy, rvv, w, h);

    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(ref[i], rvv[i])
            << label
            << " — mismatch at pixel " << i
            << " (x=" << i % w << " y=" << i / w << ")"
            << "  scalar=" << (int)ref[i]
            << "  rvv="    << (int)rvv[i]
            << "  Gx=" << Gx[i] << "  Gy=" << Gy[i];
    }

    free(ref);
    free(rvv);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Zero gradient — both paths must output all-zeros.
//    Exercises the global_max == 0 guard (clamped to 1 in the RVV impl).
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, ZeroGradient) {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    memset(Gx, 0, N * sizeof(int16_t));
    memset(Gy, 0, N * sizeof(int16_t));

    check_equiv("zero_gradient", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Uniform gradient — all raw values equal → all pixels must normalise to 255.
//    Both scalar and RVV reduction must agree on global_max.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, UniformGradient) {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = 300; Gy[i] = 400; }

    check_equiv("uniform_gradient", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Max-range gradients — exercises the overflow path (BUG 1 surface).
//    Raw L1 magnitude = 1020 + 1020 = 2040.
//    fp_scale64 * 2040 must not overflow — requires int64 multiply.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, MaxGradient) {
    const int W = 64, H = 64, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = 1020; Gy[i] = 1020; }

    check_equiv("max_gradient_1020x1020", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Negative gradients — exercises the vabs idiom (vneg + vmax).
//    Output must equal the positive-gradient result (|−x| = |x|).
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, NegativeGradients) {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = -200; Gy[i] = -150; }

    check_equiv("negative_gradients", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Mixed signs — positive and negative gradients in the same image.
//    Checks that vabs correctly handles sign diversity within one vector strip.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, MixedSigns) {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i % 2 == 0) ?  300 : -300);
        Gy[i] = static_cast<int16_t>((i % 3 == 0) ?  200 : -200);
    }

    check_equiv("mixed_signs", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Random-ish gradients — general correctness across a wide value range.
//    BUG 2 surface: small raw values after normalisation may produce -1 from
//    fixed-point rounding; vnclipu wraps -1 to 255 instead of clamping to 0.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, RandomGradients) {
    const int W = 48, H = 48, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i * 37 + 13) % 2040 - 1020);
        Gy[i] = static_cast<int16_t>((i * 53 +  7) % 2040 - 1020);
    }

    check_equiv("random_gradients_48x48", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Non-power-of-two width — exercises RVV tail/strip-mining.
//    W=100 is not a multiple of any typical VLMAX (8, 16, 32, 64),
//    forcing at least one short tail iteration where vl < VLMAX.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, NonPowerOfTwoWidth) {
    const int W = 100, H = 75, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>(i % 512 - 256);
        Gy[i] = static_cast<int16_t>((i * 3) % 512 - 256);
    }

    check_equiv("non_pot_100x75", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Single row — 1-D strip; verifies no 2-D assumptions in the RVV loop.
//    W=127 (odd prime) guarantees a tail iteration.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, SingleRow) {
    const int W = 127, H = 1, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>(i * 8 - 500);
        Gy[i] = static_cast<int16_t>(500 - i * 8);
    }

    check_equiv("single_row_127x1", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Single column — W=1, H=127.
//    Each strip has exactly one element; catches any VLMAX≥2 assumption.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, SingleColumn) {
    const int W = 1, H = 127, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>(i * 7);
        Gy[i] = 0;
    }

    check_equiv("single_column_1x127", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Single pixel — absolute minimum size.
//     Catches any assumption that n ≥ VLMAX at the start of either pass.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, SinglePixel) {
    int16_t Gx = 300, Gy = 400;
    check_equiv("single_pixel_1x1", &Gx, &Gy, 1, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. One dominant pixel — global_max is driven by a single outlier.
//     Every other pixel maps to a small value; the outlier maps to 255.
//     This is the harshest test of normalisation agreement.
//     Also re-surfaces BUG 1: fp_scale is large when global_max is small.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, OneDominantPixel) {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = 50; Gy[i] = 0; }
    Gx[N / 2] = 1020;   // single strong edge → drives normalisation scale

    check_equiv("one_dominant_pixel", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Alternating 0 / constant — every other pixel is zero.
//     After normalisation: non-zero → 255, zero → 0.
//     Tests that zeros are not corrupted by vnclipu when fixed-point yields -1.
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, AlternatingZeroAndValue) {
    const int W = 64, H = 1, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i & 1) ? 500 : 0);
        Gy[i] = 0;
    }

    check_equiv("alternating_0_500", Gx, Gy, W, H);

    free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Large image stress test — 1920×1080.
//     High pixel count increases probability of catching accumulation errors in
//     the vredmax reduction across many strips (global_max drift across VLEN
//     boundaries).
// ─────────────────────────────────────────────────────────────────────────────
TEST(MagnitudeRVV, LargeImage1080p) {
    const int W = 1920, H = 1080, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i * 17 + 3) % 2041 - 1020);
        Gy[i] = static_cast<int16_t>((i * 31 + 7) % 2041 - 1020);
    }

    check_equiv("large_1920x1080", Gx, Gy, W, H);

    free(Gx); free(Gy);
}