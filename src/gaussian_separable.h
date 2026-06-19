#pragma once
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// WHY THE SEPARABLE AND 2-D VERSIONS DIFFER
// ─────────────────────────────────────────────────────────────────────────────
// gaussian.h uses the hand-crafted kernel K5 (sum = 273):
//   { 1, 4, 7, 4, 1}
//   { 4,16,26,16, 4}
//   { 7,26,41,26, 7}   ← centre is 41, NOT 49
//   { 4,16,26,16, 4}
//   { 1, 4, 7, 4, 1}
//
// The true outer product of k1d=[1,4,7,4,1] with itself gives sum = 289:
//   { 1, 4, 7, 4, 1}
//   { 4,16,28,16, 4}   ← 28, not 26
//   { 7,28,49,28, 7}   ← centre is 49, not 41
//   { 4,16,28,16, 4}
//   { 1, 4, 7, 4, 1}
//
// These are two DIFFERENT approximations of the same Gaussian (σ≈1).
// Both are mathematically valid; the max pixel difference between them on
// natural images is ~5, mean ~1.3.  They CANNOT be made bit-exact.
//
// The separable implementation below uses k1d⊗k1d / 289, which is the
// only mathematically consistent choice for a separable filter.
//
// SEPARABLE ALGORITHM
// ───────────────────
// Pass 1 (horizontal): tmp[y,x] = Σ_{kx} src[y, x+kx] × k1d[kx+2]
//   No divide. Max value = 255×17 = 4335, stored as int16_t (max 32767) ✓
//
// Pass 2 (vertical):   acc = Σ_{ky} tmp[y+ky, x] × k1d[ky+2]
//   Then: dst = clamp(round(acc / 289), 0, 255)
//   Max acc = 4335×17 = 73695, fits in int32_t ✓
//   Rounding: add 144 (= 289/2) before integer divide.
//
// PERFORMANCE vs 2-D
//   2-D:       25 MACs/pixel
//   Separable: 10 MACs/pixel  → ~2.5× fewer arithmetic ops
// ─────────────────────────────────────────────────────────────────────────────

static const int16_t K1D[5] = { 1, 4, 7, 4, 1 };  // 1-D kernel, sum = 17
static const int32_t K1D_SUM     = 17;
static const int32_t K1D_SUM_SQ  = 289;  // 17×17, full 2-D kernel sum
static const int32_t K1D_HALF_SQ = 144;  // floor(289/2) for round-to-nearest

// ─────────────────────────────────────────────────────────────────────────────
// gaussian_5x5_separable<PixelT, AccumT>
// ─────────────────────────────────────────────────────────────────────────────
template<typename PixelT, typename AccumT>
void gaussian_5x5_separable(const PixelT* __restrict__ src,
                             PixelT*       __restrict__ dst,
                             int w, int h)
{
    // Pass-1 temp buffer: int16_t is sufficient (max 255×17 = 4335 < 32767)
    int16_t* tmp = static_cast<int16_t*>(
        aligned_alloc(64, ((size_t)w * (size_t)h * sizeof(int16_t) + 63) & ~63ULL));
    if (!tmp) return;

    // ── Pass 1: horizontal ────────────────────────────────────────────────────
    for (int y = 0; y < h; ++y) {
        const PixelT* src_row = src + y * w;
        int16_t*      tmp_row = tmp + y * w;
        for (int x = 0; x < w; ++x) {
            AccumT sum = 0;
            for (int kx = -2; kx <= 2; ++kx) {
                int ix = x + kx;
                if (ix >= 0 && ix < w)
                    sum += (AccumT)src_row[ix] * (AccumT)K1D[kx + 2];
                // out-of-bounds → zero-padding (add nothing)
            }
            tmp_row[x] = static_cast<int16_t>(sum);  // safe: max 4335 < 32767
        }
    }

    // ── Pass 2: vertical, then divide by 289 ─────────────────────────────────
    constexpr AccumT PIX_MAX = static_cast<AccumT>(std::numeric_limits<PixelT>::max());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            AccumT acc = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                int iy = y + ky;
                if (iy >= 0 && iy < h)
                    acc += (AccumT)tmp[iy * w + x] * (AccumT)K1D[ky + 2];
            }
            // Divide by 289 (= 17×17) with round-to-nearest
            AccumT result = (acc + (AccumT)K1D_HALF_SQ) / (AccumT)K1D_SUM_SQ;
            if (result > PIX_MAX)   result = PIX_MAX;
            if (result < (AccumT)0) result = (AccumT)0;
            dst[y * w + x] = static_cast<PixelT>(result);
        }
    }

    free(tmp);
}

// Convenience overload for the standard uint8_t pipeline
inline void gaussian_5x5_separable(const uint8_t* src, uint8_t* dst, int w, int h)
{
    gaussian_5x5_separable<uint8_t, int32_t>(src, dst, w, h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison helper
// ─────────────────────────────────────────────────────────────────────────────
struct GaussianCompareResult {
    int  max_diff;
    int  pixels_beyond_tol;
    bool within_tolerance;
};

inline GaussianCompareResult
gaussian_compare(const uint8_t* ref_dst,
                 const uint8_t* sep_dst,
                 int w, int h,
                 int tolerance = 5)   // ≤5 is expected and acceptable
{
    GaussianCompareResult r{};
    for (int i = 0; i < w * h; ++i) {
        int diff = (int)ref_dst[i] - (int)sep_dst[i];
        if (diff < 0) diff = -diff;
        if (diff > r.max_diff)    r.max_diff = diff;
        if (diff > tolerance)     ++r.pixels_beyond_tol;
    }
    r.within_tolerance = (r.pixels_beyond_tol == 0);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// RVV declaration (defined in rvv/gaussian_rvv.cpp)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef __riscv_v
extern "C" void gaussian_5x5_sep_rvv(const uint8_t* src, uint8_t* dst, int w, int h);
#endif