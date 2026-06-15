#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "../src/threshold.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static uint8_t* alloc_u8(int n) {
    void* p = aligned_alloc(64, ((size_t)n + 63) & ~63);
    EXPECT_NE(p, nullptr);
    return static_cast<uint8_t*>(p);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. All-zero magnitude → all EDGE_NONE
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, AllZeroMagnitude) {
    const int W = 16, H = 16, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);
    memset(out, 0xFF, N);  // poison

    threshold_and_hysteresis(mag, out, W, H, 20, 50);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(out[i], EDGE_NONE) << "Zero magnitude should produce no edges";

    free(mag); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. All-255 magnitude → all EDGE_STRONG (exceeds any reasonable high_thresh)
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, AllMaxMagnitude) {
    const int W = 16, H = 16, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 255, N);
    memset(out, 0xFF, N);

    threshold_and_hysteresis(mag, out, W, H, 50, 100);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(out[i], EDGE_STRONG)
            << "Max magnitude should produce all strong edges";

    free(mag); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. double_threshold classification (no hysteresis)
//    low=50, high=100
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, DoubleThresholdClassification) {
    const int W = 8, H = 1, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    //        0    1   2    3    4    5    6    7
    // mag:  10   40   50   80  100  150  200  255
    // exp:   0    0  128  128  255  255  255  255
    mag[0] = 10;   mag[1] = 40;   mag[2] = 50;
    mag[3] = 80;   mag[4] = 100;  mag[5] = 150;
    mag[6] = 200;  mag[7] = 255;

    double_threshold(mag, out, W, H, 50, 100);

    EXPECT_EQ(out[0], EDGE_NONE)   << "10 < low → NONE";
    EXPECT_EQ(out[1], EDGE_NONE)   << "40 < low → NONE";
    EXPECT_EQ(out[2], EDGE_WEAK)   << "50 >= low, < high → WEAK";
    EXPECT_EQ(out[3], EDGE_WEAK)   << "80 >= low, < high → WEAK";
    EXPECT_EQ(out[4], EDGE_STRONG) << "100 >= high → STRONG";
    EXPECT_EQ(out[5], EDGE_STRONG) << "150 >= high → STRONG";
    EXPECT_EQ(out[6], EDGE_STRONG) << "200 >= high → STRONG";
    EXPECT_EQ(out[7], EDGE_STRONG) << "255 >= high → STRONG";

    free(mag); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Hysteresis: weak pixel connected to strong → promoted
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, HysteresisPromotesConnectedWeak) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* img = alloc_u8(N);

    memset(img, EDGE_NONE, N);

    // Place one strong pixel at centre
    img[2 * W + 2] = EDGE_STRONG;

    // Surround with weak pixels (8-connected ring)
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (dx != 0 || dy != 0)
                img[(2 + dy) * W + (2 + dx)] = EDGE_WEAK;

    hysteresis(img, W, H);

    // All weak pixels connected to strong should now be STRONG
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            EXPECT_EQ(img[(2 + dy) * W + (2 + dx)], EDGE_STRONG)
                << "Connected weak pixel at offset (" << dx << "," << dy << ") should be promoted";

    free(img);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Hysteresis: isolated weak pixel → demoted to NONE
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, HysteresisDemotesIsolatedWeak) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* img = alloc_u8(N);

    memset(img, EDGE_NONE, N);

    // One isolated weak pixel, no strong neighbours
    img[2 * W + 2] = EDGE_WEAK;

    hysteresis(img, W, H);

    EXPECT_EQ(img[2 * W + 2], EDGE_NONE)
        << "Isolated weak pixel should be demoted to NONE";

    free(img);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Hysteresis: chain of weak pixels connected to strong
//    STRONG -- WEAK -- WEAK -- WEAK
//    All should be promoted through the chain.
//
//    NOTE: hysteresis skips border pixels (y=0, y=h-1, x=0, x=w-1) because
//    8-connected neighbourhood checks require valid neighbours on all sides.
//    We therefore use H=3 and place the chain in row 1 (the only interior row).
//    Column 0 holds EDGE_STRONG; it is on the left border but is still *read*
//    as a neighbour when processing column 1 in row 1, so chain propagation
//    works correctly.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, HysteresisChainPromotion) {
    const int W = 7, H = 3, N = W * H;  // H=3 so row 1 is a valid interior row
    uint8_t* img = alloc_u8(N);

    memset(img, EDGE_NONE, N);

    // Horizontal chain in row 1: STRONG at col 0, WEAK at cols 1-3
    img[1 * W + 0] = EDGE_STRONG;
    img[1 * W + 1] = EDGE_WEAK;
    img[1 * W + 2] = EDGE_WEAK;
    img[1 * W + 3] = EDGE_WEAK;

    hysteresis(img, W, H);

    EXPECT_EQ(img[1 * W + 0], EDGE_STRONG);
    EXPECT_EQ(img[1 * W + 1], EDGE_STRONG) << "Chain pixel 1 should be promoted";
    EXPECT_EQ(img[1 * W + 2], EDGE_STRONG) << "Chain pixel 2 should be promoted";
    EXPECT_EQ(img[1 * W + 3], EDGE_STRONG) << "Chain pixel 3 should be promoted";

    free(img);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Edge case: low_thresh == high_thresh
//    Everything >= thresh should be STRONG, everything below NONE.
//    There should be no WEAK pixels.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, EqualThresholds) {
    const int W = 8, H = 1, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    for (int i = 0; i < N; ++i)
        mag[i] = static_cast<uint8_t>(i * 32);  // 0, 32, 64, 96, 128, 160, 192, 224

    threshold_and_hysteresis(mag, out, W, H, 100, 100);

    // mag < 100  → NONE: indices 0,1,2,3  (0,32,64,96)
    // mag >= 100 → STRONG: indices 4,5,6,7 (128,160,192,224)
    EXPECT_EQ(out[0], EDGE_NONE);
    EXPECT_EQ(out[1], EDGE_NONE);
    EXPECT_EQ(out[2], EDGE_NONE);
    EXPECT_EQ(out[3], EDGE_NONE);   // 96 < 100
    EXPECT_EQ(out[4], EDGE_STRONG); // 128 >= 100
    EXPECT_EQ(out[5], EDGE_STRONG);
    EXPECT_EQ(out[6], EDGE_STRONG);
    EXPECT_EQ(out[7], EDGE_STRONG);

    free(mag); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Non-power-of-two size
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, NonPowerOfTwoSize) {
    const int W = 100, H = 75, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    for (int i = 0; i < N; ++i)
        mag[i] = static_cast<uint8_t>((i * 7 + 3) % 256);

    EXPECT_NO_FATAL_FAILURE(threshold_and_hysteresis(mag, out, W, H, 30, 80));

    free(mag); free(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Verify EDGE_STRONG (255) is never demoted by hysteresis
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, StrongNeverDemoted) {
    const int W = 5, H = 5, N = W * H;
    uint8_t* img = alloc_u8(N);

    memset(img, EDGE_STRONG, N);

    hysteresis(img, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(img[i], EDGE_STRONG)
            << "Strong pixels should never be demoted";

    free(img);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. End-to-end: threshold + hysteresis on synthetic gradient magnitude
//     Create a known pattern and verify the output is sensible.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Threshold, EndToEndSynthetic) {
    const int W = 9, H = 9, N = W * H;
    uint8_t* mag = alloc_u8(N);
    uint8_t* out = alloc_u8(N);

    memset(mag, 0, N);

    // Create a "cross" pattern: horizontal and vertical lines of high mag
    for (int x = 0; x < W; ++x) mag[4 * W + x] = 200;  // horizontal line
    for (int y = 0; y < H; ++y) mag[y * W + 4] = 200;  // vertical line

    // Some lower-magnitude neighbours
    mag[3 * W + 3] = 80;  mag[3 * W + 5] = 80;
    mag[5 * W + 3] = 80;  mag[5 * W + 5] = 80;

    threshold_and_hysteresis(mag, out, W, H, 50, 150);

    // Centre intersection and lines should be strong
    EXPECT_EQ(out[4 * W + 4], EDGE_STRONG) << "Centre should be strong";

    // Diagonal weak pixels connected to strong should be promoted
    EXPECT_EQ(out[3 * W + 3], EDGE_STRONG) << "Connected diagonal should be promoted";
    EXPECT_EQ(out[3 * W + 5], EDGE_STRONG);
    EXPECT_EQ(out[5 * W + 3], EDGE_STRONG);
    EXPECT_EQ(out[5 * W + 5], EDGE_STRONG);

    // Verify no WEAK pixels remain (all connected in this pattern)
    for (int i = 0; i < N; ++i)
        EXPECT_NE(out[i], EDGE_WEAK)
            << "No weak pixels should remain after hysteresis on this pattern";

    free(mag); free(out);
}