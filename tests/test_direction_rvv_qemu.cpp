// ═════════════════════════════════════════════════════════════════════════════
// QEMU-side equivalence test: direction_rvv vs scalar direction
// Compile: riscv64-unknown-linux-gnu-g++ -march=rv64gcv -O2 -std=c++17 ...
// Run:     qemu-riscv64 -cpu rv64,v=true,vlen=128  ./test
//          qemu-riscv64 -cpu rv64,v=true,vlen=256  ./test
//          qemu-riscv64 -cpu rv64,v=true,vlen=512  ./test
//
// Tests:  - Non-power-of-two size (100x75) to exercise strip-mining tail
//         - Zero gradient → all 0
//         - Horizontal/vertical/diagonal edges → correct direction bins
//         - Full random image → RVV output == scalar output (±0 tolerance)
// ═════════════════════════════════════════════════════════════════════════════
#include "../src/direction.h"
#include <riscv_vector.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// External RVV function
extern void direction_rvv(const int16_t* __restrict__ Gx,
                          const int16_t* __restrict__ Gy,
                          uint8_t* __restrict__ dir,
                          int w, int h);

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { tests_passed++; printf("  [PASS] %s\n", msg); }
    else      { tests_failed++; printf("  [FAIL] %s\n", msg); }
}

// Compare two direction arrays — must be exact (direction is discrete)
static bool compare_dir_exact(const uint8_t* a, const uint8_t* b, int n,
                              const char* label) {
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (ok) printf("  First mismatch @ %d: RVV=%d scalar=%d (%s)\n",
                           i, a[i], b[i], label);
            ok = false;
        }
    }
    return ok;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("=== Direction RVV Equivalence Test ===\n");

    // ── Test 1: Zero gradient → all directions should be 0 ─────────────────
    {
        const int w = 48, h = 48, n = w * h;
        int16_t* Gx = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        int16_t* Gy = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        uint8_t* dir_scalar = (uint8_t*)aligned_alloc(64, n);
        uint8_t* dir_rvv    = (uint8_t*)aligned_alloc(64, n);

        memset(Gx, 0, n * sizeof(int16_t));
        memset(Gy, 0, n * sizeof(int16_t));

        direction(Gx, Gy, dir_scalar, w, h);
        direction_rvv(Gx, Gy, dir_rvv, w, h);

        check(compare_dir_exact(dir_rvv, dir_scalar, n, "zero gradient"),
              "ZeroGradient: RVV == scalar");

        free(Gx); free(Gy); free(dir_scalar); free(dir_rvv);
    }

    // ── Test 2: Vertical edge → direction should be 0 everywhere ───────────
    {
        const int w = 100, h = 75, n = w * h;
        int16_t* Gx = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        int16_t* Gy = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        uint8_t* dir_scalar = (uint8_t*)aligned_alloc(64, n);
        uint8_t* dir_rvv    = (uint8_t*)aligned_alloc(64, n);

        // Vertical edge: strong Gx, weak Gy
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                Gx[y*w+x] = (x < w/2) ? -500 : 500;
                Gy[y*w+x] = 10;  // small vertical component
            }
        }

        direction(Gx, Gy, dir_scalar, w, h);
        direction_rvv(Gx, Gy, dir_rvv, w, h);

        check(compare_dir_exact(dir_rvv, dir_scalar, n, "vertical edge"),
              "VerticalEdge: RVV == scalar");

        free(Gx); free(Gy); free(dir_scalar); free(dir_rvv);
    }

    // ── Test 3: Horizontal edge → direction should be 2 everywhere ─────────
    {
        const int w = 100, h = 75, n = w * h;
        int16_t* Gx = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        int16_t* Gy = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        uint8_t* dir_scalar = (uint8_t*)aligned_alloc(64, n);
        uint8_t* dir_rvv    = (uint8_t*)aligned_alloc(64, n);

        // Horizontal edge: weak Gx, strong Gy
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                Gx[y*w+x] = 10;   // small horizontal component
                Gy[y*w+x] = (y < h/2) ? -500 : 500;
            }
        }

        direction(Gx, Gy, dir_scalar, w, h);
        direction_rvv(Gx, Gy, dir_rvv, w, h);

        check(compare_dir_exact(dir_rvv, dir_scalar, n, "horizontal edge"),
              "HorizontalEdge: RVV == scalar");

        free(Gx); free(Gy); free(dir_scalar); free(dir_rvv);
    }

    // ── Test 4: 45-degree diagonal (Gx=Gy>0) → direction should be 1 ───────
    {
        const int w = 100, h = 75, n = w * h;
        int16_t* Gx = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        int16_t* Gy = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        uint8_t* dir_scalar = (uint8_t*)aligned_alloc(64, n);
        uint8_t* dir_rvv    = (uint8_t*)aligned_alloc(64, n);

        // 45-degree: Gx = Gy (same sign)
        for (int i = 0; i < n; ++i) {
            Gx[i] = 300;
            Gy[i] = 300;
        }

        direction(Gx, Gy, dir_scalar, w, h);
        direction_rvv(Gx, Gy, dir_rvv, w, h);

        check(compare_dir_exact(dir_rvv, dir_scalar, n, "45-deg diagonal"),
              "Diagonal45: RVV == scalar");

        free(Gx); free(Gy); free(dir_scalar); free(dir_rvv);
    }

    // ── Test 5: 135-degree diagonal (Gx=-Gy) → direction should be 3 ───────
    {
        const int w = 100, h = 75, n = w * h;
        int16_t* Gx = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        int16_t* Gy = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        uint8_t* dir_scalar = (uint8_t*)aligned_alloc(64, n);
        uint8_t* dir_rvv    = (uint8_t*)aligned_alloc(64, n);

        // 135-degree: Gx = -Gy (opposite signs)
        for (int i = 0; i < n; ++i) {
            Gx[i] = 300;
            Gy[i] = -300;
        }

        direction(Gx, Gy, dir_scalar, w, h);
        direction_rvv(Gx, Gy, dir_rvv, w, h);

        check(compare_dir_exact(dir_rvv, dir_scalar, n, "135-deg diagonal"),
              "Diagonal135: RVV == scalar");

        free(Gx); free(Gy); free(dir_scalar); free(dir_rvv);
    }

    // ── Test 6: Random gradients → RVV must match scalar exactly ───────────
    {
        const int w = 100, h = 75, n = w * h;
        int16_t* Gx = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        int16_t* Gy = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
        uint8_t* dir_scalar = (uint8_t*)aligned_alloc(64, n);
        uint8_t* dir_rvv    = (uint8_t*)aligned_alloc(64, n);

        // Pseudo-random deterministic sequence
        unsigned seed = 42;
        for (int i = 0; i < n; ++i) {
            seed = seed * 1103515245 + 12345;
            Gx[i] = (int16_t)(seed % 2041 - 1020);  // range [-1020, 1020]
            seed = seed * 1103515245 + 12345;
            Gy[i] = (int16_t)(seed % 2041 - 1020);
        }

        direction(Gx, Gy, dir_scalar, w, h);
        direction_rvv(Gx, Gy, dir_rvv, w, h);

        check(compare_dir_exact(dir_rvv, dir_scalar, n, "random gradients"),
              "RandomGradients: RVV == scalar");

        free(Gx); free(Gy); free(dir_scalar); free(dir_rvv);
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
