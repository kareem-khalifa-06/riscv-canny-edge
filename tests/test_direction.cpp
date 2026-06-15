#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../src/direction.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static int16_t* alloc_i16(int n) {
    void* p = aligned_alloc(64, ((size_t)n * sizeof(int16_t) + 63) & ~63);
    EXPECT_NE(p, nullptr);
    return static_cast<int16_t*>(p);
}

static uint8_t* alloc_u8(int n) {
    void* p = aligned_alloc(64, ((size_t)n + 63) & ~63);
    EXPECT_NE(p, nullptr);
    return static_cast<uint8_t*>(p);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Zero gradient → direction should be 0 (Gx=0, Gy=0 falls through to first
//    branch since 0*5 < 0*2 is false and 0*5 > 0*12 is false, then diagonal
//    check: (0>=0 && 0>=0) is true → d=1)
//    Actually: ay*5 < ax*2 → 0 < 0 → false
//             ay*5 > ax*12 → 0 > 0 → false
//             else diagonal: (gx>=0 && gy>=0) || (gx<0 && gy<0)
//                            (true && true) || (false && false) = true → d=1
//    So zero gradient gives d=1 (45°).  We just verify it's consistent.
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, ZeroGradient) {
    const int W = 8, H = 8, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    memset(Gx, 0, N * sizeof(int16_t));
    memset(Gy, 0, N * sizeof(int16_t));

    direction(Gx, Gy, dir, W, H);

    // All pixels should have the SAME direction (consistent handling of 0,0)
    uint8_t first = dir[0];
    for (int i = 1; i < N; ++i)
        EXPECT_EQ(dir[i], first) << "Zero-gradient direction inconsistent at " << i;

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Pure vertical edge (strong Gx, zero Gy) → direction 0 (~0°)
//    The edge runs vertically; the gradient is horizontal.
//    ax large, ay = 0  →  ay*5 < ax*2  →  d = 0
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, VerticalEdgeIsZero) {
    const int W = 16, H = 16, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    // Strong horizontal gradient, no vertical gradient
    for (int i = 0; i < N; ++i) { Gx[i] = 500; Gy[i] = 0; }

    direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(dir[i], 0u) << "Vertical edge should have direction 0 at " << i;

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Pure horizontal edge (zero Gx, strong Gy) → direction 2 (~90°)
//    ax = 0, ay large  →  ay*5 > ax*12  →  0 > 0 is false... wait.
//    Actually: ay*5 > ax*12 → ay*5 > 0 → true if ay>0 → d=2
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, HorizontalEdgeIsNinety) {
    const int W = 16, H = 16, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    // No horizontal gradient, strong vertical gradient
    for (int i = 0; i < N; ++i) { Gx[i] = 0; Gy[i] = 500; }

    direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(dir[i], 2u) << "Horizontal edge should have direction 2 at " << i;

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. 45° diagonal (Gx > 0, Gy > 0, equal magnitude) → direction 1
//    Both gradients positive, equal magnitude → 45° diagonal
//    ay*5 vs ax*2:  500*5=2500  vs  500*2=1000  →  not < and not >
//    → diagonal branch
//    (gx>=0 && gy>=0) → true → d=1
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, Diagonal45Degrees) {
    const int W = 16, H = 16, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    for (int i = 0; i < N; ++i) { Gx[i] = 300; Gy[i] = 300; }

    direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(dir[i], 1u) << "45° diagonal should have direction 1 at " << i;

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. 135° diagonal (Gx < 0, Gy > 0, equal magnitude) → direction 3
//    (gx>=0 && gy>=0) || (gx<0 && gy<0)
//    (false && true) || (true && false) = false → d=3
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, Diagonal135Degrees) {
    const int W = 16, H = 16, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    for (int i = 0; i < N; ++i) { Gx[i] = -300; Gy[i] = 300; }

    direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(dir[i], 3u) << "135° diagonal should have direction 3 at " << i;

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. All four diagonal quadrants
//    Q1: Gx>0, Gy>0 → d=1 (45°)
//    Q2: Gx<0, Gy>0 → d=3 (135°)
//    Q3: Gx<0, Gy<0 → d=1 (45°)  [same line direction as Q1]
//    Q4: Gx>0, Gy<0 → d=3 (135°) [same line direction as Q2]
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, AllFourDiagonalQuadrants) {
    const int W = 4, H = 4, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    // Q1
    memset(Gx, 0, N * sizeof(int16_t));
    memset(Gy, 0, N * sizeof(int16_t));
    for (int i = 0; i < N; ++i) { Gx[i] = 200; Gy[i] = 200; }
    direction(Gx, Gy, dir, W, H);
    for (int i = 0; i < N; ++i) EXPECT_EQ(dir[i], 1u) << "Q1 (Gx>0,Gy>0) should be 45°";

    // Q2
    for (int i = 0; i < N; ++i) { Gx[i] = -200; Gy[i] = 200; }
    direction(Gx, Gy, dir, W, H);
    for (int i = 0; i < N; ++i) EXPECT_EQ(dir[i], 3u) << "Q2 (Gx<0,Gy>0) should be 135°";

    // Q3
    for (int i = 0; i < N; ++i) { Gx[i] = -200; Gy[i] = -200; }
    direction(Gx, Gy, dir, W, H);
    for (int i = 0; i < N; ++i) EXPECT_EQ(dir[i], 1u) << "Q3 (Gx<0,Gy<0) should be 45°";

    // Q4
    for (int i = 0; i < N; ++i) { Gx[i] = 200; Gy[i] = -200; }
    direction(Gx, Gy, dir, W, H);
    for (int i = 0; i < N; ++i) EXPECT_EQ(dir[i], 3u) << "Q4 (Gx>0,Gy<0) should be 135°";

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Non-power-of-two size
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, NonPowerOfTwoSize) {
    const int W = 100, H = 75, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    // Vertical edge pattern
    for (int i = 0; i < N; ++i) {
        Gx[i] = 500;
        Gy[i] = 0;
    }

    direction(Gx, Gy, dir, W, H);

    // All should be direction 0 (vertical edge)
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(dir[i], 0u) << "Non-PoT vertical edge failed at " << i;

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Output values are always in {0, 1, 2, 3} — never outside this set
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, OutputAlwaysInValidSet) {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    // Random-ish gradients
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i * 37 + 13) % 2000 - 1000);
        Gy[i] = static_cast<int16_t>((i * 53 + 7)  % 2000 - 1000);
    }

    direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < N; ++i) {
        EXPECT_LE(dir[i], 3u) << "Direction exceeds 3 at pixel " << i;
    }

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Nearly-horizontal gradient (small Gy, large Gx) → direction 0
//    The threshold is ay*5 < ax*2.  With Gx=500, Gy=100:
//    100*5=500 < 500*2=1000 → true → d=0
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, NearlyHorizontalGradient) {
    const int W = 8, H = 8, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    for (int i = 0; i < N; ++i) { Gx[i] = 500; Gy[i] = 100; }

    direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(dir[i], 0u) << "Nearly-horizontal gradient should be 0°";

    free(Gx); free(Gy); free(dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Nearly-vertical gradient (large Gy, small Gx) → direction 2
//     Threshold: ay*5 > ax*12.  With Gx=100, Gy=500:
//     500*5=2500 > 100*12=1200 → true → d=2
// ─────────────────────────────────────────────────────────────────────────────
TEST(Direction, NearlyVerticalGradient) {
    const int W = 8, H = 8, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    uint8_t* dir = alloc_u8(N);

    for (int i = 0; i < N; ++i) { Gx[i] = 100; Gy[i] = 500; }

    direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(dir[i], 2u) << "Nearly-vertical gradient should be 90°";

    free(Gx); free(Gy); free(dir);
}
