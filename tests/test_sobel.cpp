#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../src/sobel.h"

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

// ─────────────────────────────────────────────────────────────────────────────
// 1. Uniform image → Gx = 0, Gy = 0 everywhere
//    Any constant image has zero gradient by definition.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, UniformImageZeroGradient) {
    const int W = 32, H = 32;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    memset(src, 128, W * H);

    sobel(src, Gx, Gy, W, H);

    // Interior only — border pixels touch zero-padded region so may be non-zero
    for (int y = 1; y < H - 1; ++y)
        for (int x = 1; x < W - 1; ++x) {
            EXPECT_EQ(Gx[y * W + x], 0) << "Gx non-zero at (" << x << "," << y << ")";
            EXPECT_EQ(Gy[y * W + x], 0) << "Gy non-zero at (" << x << "," << y << ")";
        }

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Sharp vertical edge (left=0, right=255):
//    Large |Gx| at the edge column, Gy ≈ 0 in the interior rows.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, VerticalEdgeLargeGxSmallGy) {
    const int W = 32, H = 32;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x < W / 2) ? 0 : 255;

    sobel(src, Gx, Gy, W, H);

    // At the edge column (W/2), interior rows: |Gx| must be large
    int edge_x = W / 2;
    for (int y = 1; y < H - 1; ++y) {
        EXPECT_GT(std::abs(Gx[y * W + edge_x]), 500)
            << "Expected large |Gx| at vertical edge, row " << y;
        EXPECT_EQ(Gy[y * W + edge_x], 0)
            << "Expected zero Gy at vertical edge interior, row " << y;
    }

    // Far from the edge: both gradients should be zero
    for (int y = 1; y < H - 1; ++y) {
        EXPECT_EQ(Gx[y * W + 2], 0) << "Gx should be 0 far left of edge";
        EXPECT_EQ(Gx[y * W + (W - 3)], 0) << "Gx should be 0 far right of edge";
    }

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Sharp horizontal edge (top=0, bottom=255):
//    Large |Gy| at the edge row, Gx ≈ 0 in the interior columns.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, HorizontalEdgeLargeGySmallGx) {
    const int W = 32, H = 32;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (y < H / 2) ? 0 : 255;

    sobel(src, Gx, Gy, W, H);

    int edge_y = H / 2;
    for (int x = 1; x < W - 1; ++x) {
        EXPECT_GT(std::abs(Gy[edge_y * W + x]), 500)
            << "Expected large |Gy| at horizontal edge, col " << x;
        EXPECT_EQ(Gx[edge_y * W + x], 0)
            << "Expected zero Gx at horizontal edge interior, col " << x;
    }

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Diagonal edge → both Gx and Gy are non-zero near the diagonal
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, DiagonalEdgeBothGradientsNonZero) {
    const int W = 32, H = 32;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x > y) ? 255 : 0;

    sobel(src, Gx, Gy, W, H);

    // Sample a few pixels on the diagonal (interior only)
    bool found_both = false;
    for (int d = 2; d < W - 2; ++d) {
        if (std::abs(Gx[d * W + d]) > 0 && std::abs(Gy[d * W + d]) > 0) {
            found_both = true;
            break;
        }
    }
    EXPECT_TRUE(found_both) << "Diagonal edge should produce non-zero Gx AND Gy";

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. All-black image → all gradients zero
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, AllBlackZeroGradient) {
    const int W = 64, H = 64;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    memset(src, 0, W * H);
    // Poison outputs to detect if they are not written
    memset(Gx, 0x7F, W * H * sizeof(int16_t));
    memset(Gy, 0x7F, W * H * sizeof(int16_t));

    sobel(src, Gx, Gy, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_EQ(Gx[i], 0) << "Gx non-zero on all-black at pixel " << i;
        EXPECT_EQ(Gy[i], 0) << "Gy non-zero on all-black at pixel " << i;
    }

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Output range check: int16_t can hold max Sobel value
//    Max possible: 4 * 255 = 1020, which fits int16_t (max 32767)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, OutputFitsInInt16) {
    const int W = 32, H = 32;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    // Worst case: checkerboard maximises local contrast
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = ((x + y) & 1) ? 255 : 0;

    sobel(src, Gx, Gy, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_LE(std::abs((int)Gx[i]), 1020) << "Gx overflow at pixel " << i;
        EXPECT_LE(std::abs((int)Gy[i]), 1020) << "Gy overflow at pixel " << i;
    }

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. SoA layout: Gx and Gy are separate arrays (not interleaved)
//    Verify independent write by checking they can differ at the same index.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, SoALayoutGxGyAreIndependent) {
    const int W = 16, H = 16;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    // Vertical edge → Gx large, Gy=0; confirms arrays are truly separate
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x < W / 2) ? 0 : 255;

    sobel(src, Gx, Gy, W, H);

    // At the edge, Gx != 0 but Gy == 0 — proves independent storage
    int edge_x = W / 2;
    for (int y = 1; y < H - 1; ++y) {
        EXPECT_NE(Gx[y * W + edge_x], 0) << "Gx should be non-zero at edge";
        EXPECT_EQ(Gy[y * W + edge_x], 0) << "Gy should be zero at vertical edge";
    }

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Zero-padding boundary: border pixels must not read out-of-bounds
//    (this test just checks the function runs without crash/sanitizer error)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, BorderPixelsNoOutOfBoundsAccess) {
    const int W = 8, H = 8;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    for (int i = 0; i < W * H; ++i) src[i] = static_cast<uint8_t>(i * 7);

    EXPECT_NO_FATAL_FAILURE(sobel(src, Gx, Gy, W, H));

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Non-power-of-two size (strip-mining readiness)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, NonPowerOfTwoSize) {
    const int W = 100, H = 75;
    uint8_t*  src = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    // Vertical edge at centre
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x < W / 2) ? 0 : 255;

    EXPECT_NO_FATAL_FAILURE(sobel(src, Gx, Gy, W, H));

    // Spot-check: gradient at edge column should still be large
    int edge_x = W / 2;
    EXPECT_GT(std::abs(Gx[H / 2 * W + edge_x]), 500)
        << "Large Gx expected at vertical edge for non-PoT size";

    free(src); free(Gx); free(Gy);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Antisymmetry: flipping the edge direction negates the gradient
//     If left=255,right=0  →  Gx should be opposite sign vs left=0,right=255
// ─────────────────────────────────────────────────────────────────────────────
TEST(Sobel, GradientSignFlipsWithEdgeDirection) {
    const int W = 32, H = 32;
    uint8_t*  src_a = alloc_u8(W * H);
    uint8_t*  src_b = alloc_u8(W * H);
    int16_t*  Gx_a  = alloc_i16(W * H);
    int16_t*  Gy_a  = alloc_i16(W * H);
    int16_t*  Gx_b  = alloc_i16(W * H);
    int16_t*  Gy_b  = alloc_i16(W * H);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            src_a[y * W + x] = (x < W / 2) ? 0   : 255;  // dark→bright
            src_b[y * W + x] = (x < W / 2) ? 255 : 0;    // bright→dark
        }

    sobel(src_a, Gx_a, Gy_a, W, H);
    sobel(src_b, Gx_b, Gy_b, W, H);

    int edge_x = W / 2;
    int row    = H / 2;
    // Signs must be opposite
    EXPECT_GT(Gx_a[row * W + edge_x], 0)  << "src_a should have positive Gx";
    EXPECT_LT(Gx_b[row * W + edge_x], 0)  << "src_b should have negative Gx";
    // Magnitudes must be equal
    EXPECT_EQ(std::abs(Gx_a[row * W + edge_x]),
              std::abs(Gx_b[row * W + edge_x]))
        << "Magnitude should be identical for mirrored edges";

    free(src_a); free(src_b);
    free(Gx_a);  free(Gy_a);
    free(Gx_b);  free(Gy_b);
}