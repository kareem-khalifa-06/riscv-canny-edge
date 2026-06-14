
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "../src/magnitude.h"

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static int16_t* alloc_i16(int n) {
    void* p = aligned_alloc(64, ((size_t)n * sizeof(int16_t) + 63) & ~63);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return static_cast<int16_t*>(p);
}

static uint8_t* alloc_u8(int n) {
    void* p = aligned_alloc(64, ((size_t)n + 63) & ~63);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return static_cast<uint8_t*>(p);
}

// Core comparison: run both implementations, assert pixel-exact agreement.
// Returns the number of mismatches (0 = pass).
static int compare(const char* label,
                   int16_t* Gx, int16_t* Gy,
                   int w, int h)
{
    int n = w * h;
    uint8_t* ref = alloc_u8(n);
    uint8_t* rvv = alloc_u8(n);

    // Poison outputs to detect unwritten pixels
    memset(ref, 0xAA, n);
    memset(rvv, 0x55, n);

    magnitude_l1    (Gx, Gy, ref, w, h);
    magnitude_l1_rvv(Gx, Gy, rvv, w, h);

    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (ref[i] != rvv[i]) {
            if (mismatches < 8)   // cap noise in output
                fprintf(stderr,
                    "  [%s] MISMATCH pixel %d (x=%d y=%d): "
                    "scalar=%u rvv=%u  Gx=%d Gy=%d\n",
                    label, i, i % w, i / w,
                    ref[i], rvv[i], Gx[i], Gy[i]);
            ++mismatches;
        }
    }

    if (mismatches == 0)
        printf("PASS  %-40s  %dx%d\n", label, w, h);
    else
        printf("FAIL  %-40s  %dx%d  (%d mismatches)\n",
               label, w, h, mismatches);

    free(ref);
    free(rvv);
    return mismatches;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test cases
// ─────────────────────────────────────────────────────────────────────────────

// T1: Zero gradient — both must output all-zeros.
//     Exercises the global_max == 0 guard (set to 1).
static int t_zero_gradient() {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    memset(Gx, 0, N * sizeof(int16_t));
    memset(Gy, 0, N * sizeof(int16_t));
    int r = compare("zero_gradient", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T2: Uniform gradient — all raw values equal → normalise to 255.
//     Both implementations should produce all-255 output.
static int t_uniform_gradient() {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = 300; Gy[i] = 400; }
    int r = compare("uniform_gradient_300_400", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T3: Max-range gradients — exercises overflow / clamping path.
//     BUG 1 surface: fp_scale * raw_max may overflow int32_t here.
static int t_max_gradient() {
    const int W = 64, H = 64, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = 1020; Gy[i] = 1020; }
    int r = compare("max_gradient_1020_1020", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T4: Negative gradients — exercises vabs idiom (vneg + vmax).
static int t_negative_gradient() {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = -200; Gy[i] = -150; }
    int r = compare("negative_gradient_-200_-150", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T5: Mixed signs — both positive and negative gradients in the same image.
static int t_mixed_signs() {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i % 2 == 0) ?  300 : -300);
        Gy[i] = static_cast<int16_t>((i % 3 == 0) ?  200 : -200);
    }
    int r = compare("mixed_signs", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T6: General random-ish gradients — wide spread exercises the normalisation
//     scale; also exposes BUG 2 (negative fixed-point residuals → vnclipu wrap).
static int t_random_gradients() {
    const int W = 48, H = 48, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i * 37 + 13) % 2040 - 1020);
        Gy[i] = static_cast<int16_t>((i * 53 +  7) % 2040 - 1020);
    }
    int r = compare("random_gradients_48x48", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T7: Non-power-of-two width — exercises RVV tail/strip-mining loop.
//     W=100 is not a multiple of any typical VLMAX, forcing a tail iteration.
static int t_non_pot_width() {
    const int W = 100, H = 75, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>(i % 512 - 256);
        Gy[i] = static_cast<int16_t>((i * 3) % 512 - 256);
    }
    int r = compare("non_pot_100x75", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T8: Width = 1 — single column; every pixel is a separate vl=1 iteration.
//     Catches off-by-one in strip advancement (i += vl must be exact).
static int t_single_column() {
    const int W = 1, H = 127, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>(i * 7);
        Gy[i] = 0;
    }
    int r = compare("single_column_1x127", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T9: Width = 1, Height = 1 — absolute minimum; guard for any VLMAX assumption.
static int t_single_pixel() {
    int16_t Gx = 300, Gy = 400;
    return compare("single_pixel_1x1", &Gx, &Gy, 1, 1);
}

// T10: One dominant pixel — the normalisation divisor (global_max) is driven by
//      a single outlier; every other pixel should map to a low value.
//      This is the sharpest test of the normalisation path agreement.
//      Also exercises BUG 1: fp_scale is large when global_max is small.
static int t_one_dominant_pixel() {
    const int W = 32, H = 32, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) { Gx[i] = 50; Gy[i] = 0; }
    Gx[N / 2] = 1020;   // single strong edge drives the normalisation scale
    int r = compare("one_dominant_pixel", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T11: Alternating 0 / max — exercises every other pixel being zero.
//      After normalisation non-zero pixels → 255, zero pixels → 0.
static int t_alternating() {
    const int W = 64, H = 1, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i & 1) ? 500 : 0);
        Gy[i] = 0;
    }
    int r = compare("alternating_0_500_1x64", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// T12: Large image — stress test; increases probability of catching latent
//      accumulation errors in the reduction (vredmax across many strips).
static int t_large_image() {
    const int W = 1920, H = 1080, N = W * H;
    int16_t* Gx = alloc_i16(N);
    int16_t* Gy = alloc_i16(N);
    for (int i = 0; i < N; ++i) {
        Gx[i] = static_cast<int16_t>((i * 17 + 3) % 2041 - 1020);
        Gy[i] = static_cast<int16_t>((i * 31 + 7) % 2041 - 1020);
    }
    int r = compare("large_1920x1080", Gx, Gy, W, H);
    free(Gx); free(Gy);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== magnitude_l1_rvv equivalence tests (scalar vs RVV) ===\n\n");

    int total_fail = 0;

    total_fail += t_zero_gradient();
    total_fail += t_uniform_gradient();
    total_fail += t_max_gradient();
    total_fail += t_negative_gradient();
    total_fail += t_mixed_signs();
    total_fail += t_random_gradients();
    total_fail += t_non_pot_width();
    total_fail += t_single_column();
    total_fail += t_single_pixel();
    total_fail += t_one_dominant_pixel();
    total_fail += t_alternating();
    total_fail += t_large_image();

    printf("\n");
    if (total_fail == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("FAILED — %d total pixel mismatches across all tests\n", total_fail);

    return total_fail ? 1 : 0;
}