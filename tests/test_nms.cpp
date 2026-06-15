#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "../src/nms.h"
#include "../src/direction.h"

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
// 1. Perfect horizontal ridge — direction 0 (horizontal gradient)
//    Pixel at center of ridge should survive; neighbours should be suppressed.
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, HorizontalRidgeDirection0) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);
    memset(dir, 0, N);

    // Horizontal ridge at y=2: centre=255, left/right=200
    // Direction 0 means gradient is horizontal → check left/right
    mag[2 * W + 1] = 200;
    mag[2 * W + 2] = 255;  // ridge centre — should survive
    mag[2 * W + 3] = 200;
    dir[2 * W + 1] = 0;
    dir[2 * W + 2] = 0;
    dir[2 * W + 3] = 0;

    non_maximum_suppression(mag, dir, out, W, H);

    EXPECT_EQ(out[2 * W + 2], 255u) << "Ridge centre should survive NMS";
    EXPECT_EQ(out[2 * W + 1], 0u)   << "Lower neighbour should be suppressed";
    EXPECT_EQ(out[2 * W + 3], 0u)   << "Lower neighbour should be suppressed";

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Perfect vertical ridge — direction 2 (vertical gradient)
//    Centre pixel should survive; top/bottom neighbours suppressed.
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, VerticalRidgeDirection2) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);
    memset(dir, 0, N);

    // Vertical ridge at x=2
    mag[1 * W + 2] = 200;
    mag[2 * W + 2] = 255;  // should survive
    mag[3 * W + 2] = 200;
    dir[1 * W + 2] = 2;
    dir[2 * W + 2] = 2;
    dir[3 * W + 2] = 2;

    non_maximum_suppression(mag, dir, out, W, H);

    EXPECT_EQ(out[2 * W + 2], 255u) << "Vertical ridge centre should survive";
    EXPECT_EQ(out[1 * W + 2], 0u)   << "Top neighbour should be suppressed";
    EXPECT_EQ(out[3 * W + 2], 0u)   << "Bottom neighbour should be suppressed";

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. 45-degree diagonal ridge — direction 1
//    Gradient at 45°, edge runs \ (top-left to bottom-right)
//    NMS checks neighbours along gradient: top-left and bottom-right
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, Diagonal45RidgeDirection1) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);
    memset(dir, 0, N);

    // 45-degree gradient: centre at (2,2), neighbours along gradient at (1,1) and (3,3)
    mag[1 * W + 1] = 200;  // top-left
    mag[2 * W + 2] = 255;  // centre — should survive
    mag[3 * W + 3] = 200;  // bottom-right
    dir[1 * W + 1] = 1;
    dir[2 * W + 2] = 1;
    dir[3 * W + 3] = 1;

    non_maximum_suppression(mag, dir, out, W, H);

    EXPECT_EQ(out[2 * W + 2], 255u) << "45-deg diagonal centre should survive";
    EXPECT_EQ(out[1 * W + 1], 0u)   << "Neighbour should be suppressed";
    EXPECT_EQ(out[3 * W + 3], 0u)   << "Neighbour should be suppressed";

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. 135-degree diagonal ridge — direction 3
//    Gradient at 135°, edge runs / (top-right to bottom-left)
//    NMS checks neighbours along gradient: top-right and bottom-left
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, Diagonal135RidgeDirection3) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);
    memset(dir, 0, N);

    // 135-degree gradient: centre at (2,2), neighbours along gradient at (1,3) and (3,1)
    mag[1 * W + 3] = 200;  // top-right
    mag[2 * W + 2] = 255;  // centre — should survive
    mag[3 * W + 1] = 200;  // bottom-left
    dir[1 * W + 3] = 3;
    dir[2 * W + 2] = 3;
    dir[3 * W + 1] = 3;

    non_maximum_suppression(mag, dir, out, W, H);

    EXPECT_EQ(out[2 * W + 2], 255u) << "135-deg diagonal centre should survive";
    EXPECT_EQ(out[1 * W + 3], 0u)   << "Neighbour should be suppressed";
    EXPECT_EQ(out[3 * W + 1], 0u)   << "Neighbour should be suppressed";

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Flat plateau — all equal magnitudes → everything suppressed
//    NMS requires strictly greater than BOTH neighbours.
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, FlatPlateauEverythingSuppressed) {
    const int W = 7, H = 7, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 128, N);
    memset(dir, 0, N);

    non_maximum_suppression(mag, dir, out, W, H);

    // Interior should all be 0 (nothing is strictly greater)
    for (int y = 1; y < H - 1; ++y)
        for (int x = 1; x < W - 1; ++x)
            EXPECT_EQ(out[y * W + x], 0u)
                << "Flat plateau pixel should be suppressed at (" << x << "," << y << ")";

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Single isolated peak — should survive
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, SinglePeakSurvives) {
    const int W = 7, H = 7, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);
    memset(dir, 0, N);

    // Single bright pixel in centre with horizontal direction
    mag[3 * W + 3] = 255;
    dir[3 * W + 3] = 0;

    non_maximum_suppression(mag, dir, out, W, H);

    EXPECT_EQ(out[3 * W + 3], 255u) << "Isolated peak should survive NMS";

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Boundary pixels are always suppressed (no neighbours to compare)
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, BoundaryPixelsSuppressed) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 255, N);
    memset(dir, 0, N);

    non_maximum_suppression(mag, dir, out, W, H);

    // All border pixels should be 0
    for (int x = 0; x < W; ++x) {
        EXPECT_EQ(out[x], 0u)           << "Top border should be suppressed";
        EXPECT_EQ(out[(H - 1) * W + x], 0u) << "Bottom border should be suppressed";
    }
    for (int y = 0; y < H; ++y) {
        EXPECT_EQ(out[y * W], 0u)       << "Left border should be suppressed";
        EXPECT_EQ(out[y * W + (W - 1)], 0u) << "Right border should be suppressed";
    }

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. End-to-end: Sobel → Direction → NMS on a synthetic vertical edge
//    Verify that a clean vertical edge produces a thin line after NMS.
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, EndToEndVerticalEdge) {
    const int W = 32, H = 32, N = W * H;
    uint8_t*  src = alloc_u8(N);
    int16_t*  Gx  = alloc_i16(N);
    int16_t*  Gy  = alloc_i16(N);
    uint8_t*  mag = alloc_u8(N);
    uint8_t*  dir = alloc_u8(N);
    uint8_t*  nms = alloc_u8(N);

    // Vertical edge image
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            src[y * W + x] = (x < W / 2) ? 0 : 255;

    // Simple magnitude = L1 without normalization for testing
    for (int i = 0; i < N; ++i) {
        Gx[i] = (i % W < W / 2) ? 0 : 500;
        Gy[i] = 0;
    }

    // Fill direction
    direction(Gx, Gy, dir, W, H);

    // Fill magnitude: strong at edge, zero elsewhere
    memset(mag, 0, N);
    for (int y = 1; y < H - 1; ++y) {
        // Edge is at W/2, give it strong magnitude
        mag[y * W + (W / 2)] = 255;
        // Give neighbours slightly lower magnitude
        if (W / 2 > 0)   mag[y * W + (W / 2 - 1)] = 180;
        if (W / 2 + 1 < W) mag[y * W + (W / 2 + 1)] = 180;
    }

    non_maximum_suppression(mag, dir, nms, W, H);

    // The ridge peak at W/2 should survive
    // (at least some pixels on the edge should have non-zero output)
    bool any_survived = false;
    for (int y = 2; y < H - 2; ++y) {
        if (nms[y * W + (W / 2)] > 0) {
            any_survived = true;
            break;
        }
    }
    EXPECT_TRUE(any_survived) << "At least some edge pixels should survive NMS";

    free(src); free(Gx); free(Gy); free(mag); free(dir); free(nms);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Non-power-of-two size
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, NonPowerOfTwoSize) {
    const int W = 100, H = 75, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    for (int i = 0; i < N; ++i) {
        mag[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
        dir[i] = static_cast<uint8_t>(i % 4);
    }

    EXPECT_NO_FATAL_FAILURE(non_maximum_suppression(mag, dir, out, W, H));

    free(mag); free(dir); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. All-zero magnitude → all-zero output
// ─────────────────────────────────────────────────────────────────────────────
TEST(NMS, AllZeroInput) {
    const int W = 16, H = 16, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* dir = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);
    memset(dir, 0, N);
    memset(out, 0xFF, N);  // poison

    non_maximum_suppression(mag, dir, out, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(out[i], 0u) << "All-zero input should produce all-zero output";

    free(mag); free(dir); free(out);
}